"""Recompute manuscript summaries and figures from the included score tables.

Usage: python scripts/reproduce_paper.py --check
This does not refit neural predictions. See run_predictions.py for that step.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BUDGETS = np.array([1, 2, 4, 5])
LABELS = {
    "D1": "Direct", "D1_SHRUNK": "Low-rank direct reference",
    "D1_PAIR_CALIBRATED": "Mapped", "D1_DXY": "Mapped + position",
    "F1": "Visual profile", "S1": "Handcrafted visual profile",
    "A1": "Activity profile", "R2": "Anchor profile", "FA1": "Visual-activity hybrid",
}
SCORES = ["r_B", "r_BG", "r_BR_b", "r_BRG_b", "r_BR_hi", "r_BRG_hi"]


def aulc(values):
    """Normalized trapezoidal area on log2 repeat budget."""
    values = np.asarray(values, dtype=float)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError("AULC requires four finite values at budgets 1, 2, 4, 5")
    x = np.log2(BUDGETS)
    return float(np.sum(np.diff(x) * (values[:-1] + values[1:]) / 2) / x[-1])


def inverse(score, reference, scale="log"):
    """Interpolate only inside a strictly increasing measured reference."""
    reference = np.asarray(reference, dtype=float)
    if reference.shape != (4,) or not np.isfinite(reference).all() or np.any(np.diff(reference) <= 0):
        raise ValueError("Reference must have four finite, strictly increasing scores")
    if not np.isfinite(score) or score < reference[0] or score > reference[-1]:
        return np.nan
    if scale not in ("log", "linear"):
        raise ValueError("scale must be log or linear")
    x = np.log2(BUDGETS) if scale == "log" else BUDGETS
    value = np.interp(score, reference, x)
    return float(2 ** value if scale == "log" else value)


def aggregate(cells, key, value, movie="test_movie", orientation="repeat_orientation", subset="calibration_subset_id"):
    """Equal means over subsets, allocations, folds, movies, then scans.

    Targets have already been averaged within each technical cell. Do not
    weight cell means by target count: that changes the paper's estimand.
    """
    key = list(key)
    identity = key + ["scan", movie, orientation, "outer_fold", subset]
    if cells.duplicated(identity).any():
        raise ValueError("Duplicate technical score cells")
    values = [value] if isinstance(value, str) else list(value)
    if not np.isfinite(cells[values].to_numpy()).all():
        raise ValueError("Nonfinite cell scores must not be silently dropped")
    groups = key + ["scan", movie, orientation, "outer_fold"]
    current = cells.groupby(groups, as_index=False)[values].mean()
    for remove in [orientation, "outer_fold", movie]:
        groups.remove(remove)
        current = current.groupby(groups, as_index=False)[values].mean()
    return current, current.groupby(key, as_index=False)[values].mean()


def curve(frame, candidate, policy="selected"):
    rows = frame[(frame.candidate == candidate) & (frame.policy_arm == policy)]
    if rows.budget.duplicated().any() or set(rows.budget) != set(BUDGETS):
        raise ValueError(f"Incomplete curve for {candidate}/{policy}")
    return rows.set_index("budget").loc[BUDGETS, "mean_target_pearson"].to_numpy()


def location_summary(cells):
    scan, overall = aggregate(cells, ["budget"], SCORES)
    rows = []
    for budget, data in cells.groupby("budget"):
        # Keep movies while averaging technical configurations and scans.
        movie_scan = data.groupby(["scan", "test_movie", "repeat_orientation", "outer_fold"], as_index=False)[SCORES].mean()
        movie_scan = movie_scan.groupby(["scan", "test_movie", "outer_fold"], as_index=False)[SCORES].mean()
        movie_scan = movie_scan.groupby(["scan", "test_movie"], as_index=False)[SCORES].mean()
        movies = movie_scan.groupby("test_movie")[SCORES].mean().sort_index()
        if len(movies) != 6:
            raise ValueError("Movie reweighting requires all six observed movie identities")
        weights = np.random.default_rng(20260910).multinomial(6, np.full(6, 1 / 6), size=10000)
        draws = pd.DataFrame(weights @ movies.to_numpy() / 6, columns=SCORES)
        means = overall.set_index("budget").loc[budget]
        contrasts = [("None", "Position", "r_BG", "r_B")]
        for regime, suffix in [("Budget-matched", "b"), ("Five repeats", "hi")]:
            contrasts.extend([(regime, "Location", f"r_BR_{suffix}", "r_B"),
                              (regime, "Position after location", f"r_BRG_{suffix}", f"r_BR_{suffix}")])
        for regime, feature, high, low in contrasts:
            limits = np.quantile(draws[high] - draws[low], [.025, .975], method="linear")
            rows.append(dict(budget=budget, location_fit=regime, added_feature=feature,
                             gain=float(means[high] - means[low]), lower=limits[0], upper=limits[1]))
    return pd.DataFrame(rows), scan


def save_table(frame, out, name):
    frame.to_csv(out / f"{name}.csv", index=False, float_format="%.15g")


def build_tables(data, out):
    spatial_cells = pd.read_csv(data / "spatial/cells.csv.gz", float_precision="round_trip")
    spatial_scan, spatial = aggregate(spatial_cells, ["candidate", "policy_arm", "budget"], "mean_target_pearson")
    profile_cells = pd.read_csv(data / "profiles/cells.csv.gz", float_precision="round_trip")
    _, profiles = aggregate(profile_cells, ["candidate", "policy_arm", "budget"], "mean_target_pearson")
    reference = curve(profiles, "D1_SHRUNK")
    if not np.isfinite(reference).all():
        raise ValueError("Reference is incomplete")
    curves = {name: curve(spatial if name.startswith("D1") else profiles, name)
              for name in LABELS if name != "D1_SHRUNK"}
    curves["D1_SHRUNK"] = reference
    table8 = pd.DataFrame([dict(predictor=LABELS[name], **{f"b{b}": v for b, v in zip(BUDGETS, curves[name])}, aulc=aulc(curves[name])) for name in LABELS])
    table1 = table8[table8.predictor.isin([LABELS[n] for n in ["D1", "D1_PAIR_CALIBRATED", "D1_DXY"]])].copy().reset_index(drop=True)
    for label, high, low in [("Mapping gain", "D1_PAIR_CALIBRATED", "D1"), ("Position gain", "D1_DXY", "D1_PAIR_CALIBRATED")]:
        delta = curves[high] - curves[low]
        table1.loc[len(table1)] = [label, *delta, aulc(delta)]
    table4 = []
    for label, ref, scale in [("Low-rank direct", reference, "log"), ("Low-rank direct", reference, "linear"), ("Direct", curves["D1"], "log"), ("Mapped", curves["D1_PAIR_CALIBRATED"], "log")]:
        prices = [inverse(a, ref, scale) - inverse(b, ref, scale) for a, b in zip(curves["D1_DXY"], curves["D1_PAIR_CALIBRATED"])]
        table4.append(dict(reference=label, interpolation=scale, **{f"b{b}": v for b, v in zip(BUDGETS, prices)}))
    comparisons = [("In-plane distance", "D1_DXY", "D1_PAIR_CALIBRATED"),
        ("Released depth-coordinate separation", "D1_DZ", "D1_PAIR_CALIBRATED"),
        ("Three-dimensional distance", "D1_D3D", "D1_PAIR_CALIBRATED"),
        ("Joint depth and in-plane separation", "D1_DZ_DXY", "D1_PAIR_CALIBRATED"),
        ("Depth separation after in-plane distance", "D1_DZ_DXY", "D1_DXY"),
        ("In-plane distance after depth separation", "D1_DZ_DXY", "D1_DZ"),
        ("Three-dimensional minus in-plane distance", "D1_D3D", "D1_DXY")]
    spatial_deltas = []
    for label, high, low in comparisons:
        delta = curve(spatial, high) - curve(spatial, low)
        spatial_deltas.append(dict(contrast=label, **{f"b{b}": v for b, v in zip(BUDGETS, delta)}, aulc=aulc(delta)))
    table5 = pd.DataFrame(spatial_deltas)
    within = curve(spatial, "WF_D1_DXY") - curve(spatial, "WF_D1_PAIR_CALIBRATED")
    locations, _ = location_summary(pd.read_csv(data / "location/cells.csv.gz", float_precision="round_trip"))
    anatomy_cells = pd.read_csv(data / "anatomy/cells.csv.gz", float_precision="round_trip")
    anatomy_scan, anatomy = aggregate(anatomy_cells, ["panel", "candidate", "policy", "budget"], "value", movie="movie_rotation", orientation="orientation", subset="subset")
    anatomy_rows = []
    for (panel, candidate, policy), group in anatomy.groupby(["panel", "candidate", "policy"]):
        values = group.set_index("budget").value.reindex(BUDGETS).to_numpy()
        anatomy_rows.append(dict(panel=panel, candidate=candidate, policy=policy, **{f"b{b}": v for b,v in zip(BUDGETS, values)}, aulc=aulc(values)))
    anatomy_endpoints = pd.DataFrame(anatomy_rows)
    selected_anatomy = anatomy_endpoints.query("panel == 'anatomy' and policy == 'selected'").set_index("candidate")
    table7_rows = []
    for name, label in [("B","Mapped correlations"), ("BG","Mapped + position"), ("BT","Mapped + cell class"),
                        ("BGL","Mapped + position + coarse class"), ("BGT","Mapped + position + detailed class")]:
        row = selected_anatomy.loc[name]
        table7_rows.append(dict(predictor_or_contrast=label, b1=row.b1, aulc=row.aulc))
    for label, high, low in [("Position beyond mapped correlations","BG","B"), ("Position after cell class","BGT","BT"),
                             ("Cell class after position","BGT","BG"), ("Detailed beyond coarse class, with position","BGT","BGL")]:
        difference=selected_anatomy.loc[high,["b1","aulc"]]-selected_anatomy.loc[low,["b1","aulc"]]
        table7_rows.append(dict(predictor_or_contrast=label, b1=difference.b1, aulc=difference.aulc))
    table7=pd.DataFrame(table7_rows)
    for name, table in [("table1_prediction_and_gains", table1), ("table4_reference_prices", pd.DataFrame(table4)),
        ("table5_spatial_features", table5), ("table6_location_controls", locations),
        ("table7_anatomical_comparisons", table7), ("anatomy_all_candidates", anatomy_endpoints), ("table8_complete_pipelines", table8),
        ("spatial_scan_curves", spatial_scan), ("anatomy_scan_curves", anatomy_scan)]:
        save_table(table, out, name)
    quality = pd.read_csv(data / "location/quality_scan.csv", float_precision="round_trip").groupby("rf_budget", as_index=False).mean(numeric_only=True)
    table2 = locations[locations.budget.eq(1) & locations.added_feature.str.startswith("Position")].copy()
    dispersion = quality.set_index("rf_budget").median_leave_one_fold_out_rms_grid
    table2["location_dispersion"] = table2.location_fit.map({"Budget-matched":dispersion.loc[1], "Five repeats":dispersion.loc[5]})
    save_table(table2, out, "table2_location_controls")
    save_table(quality, out, "location_fit_quality")
    save_table(pd.DataFrame({"budget": BUDGETS, "gain": within}), out, "within_field")
    delta = curves["D1_DXY"] - curves["D1_PAIR_CALIBRATED"]
    return dict(curves=curves, spatial=spatial, scan=spatial_scan, spatial_features=table5, locations=locations,
                anatomy=anatomy_endpoints, table7=table7, prices=pd.DataFrame(table4), within=within,
                overstatement_percent=100 * (curves["D1_PAIR_CALIBRATED"][0] - curves["D1"][0]) / delta[0])


def check_manuscript(result):
    """Check canonical manuscript values at their stated numerical precision."""
    expected = {
        "D1": [.212008, .254257, .295786, .307099],
        "D1_PAIR_CALIBRATED": [.225955, .257702, .294075, .304795],
        "D1_DXY": [.256668, .278148, .305641, .314290],
        "F1": [.2465, .2478, .2554, .2583], "S1": [.2293, .2428, .2573, .2612],
        "A1": [.2575, .2788, .3049, .3130], "R2": [.2431, .2448, .2461, .2464],
        "FA1": [.2574, .2784, .3043, .3122],
    }
    for name, values in expected.items():
        np.testing.assert_allclose(result["curves"][name], values, atol=5.1e-7 if name.startswith("D1") else 5.1e-5, rtol=0, err_msg=name)
    np.testing.assert_allclose(result["prices"].iloc[0][["b1", "b2", "b4"]].to_numpy(dtype=float), [.824625, .861554, .973805], atol=5.1e-7, rtol=0)
    if not result["prices"].b5.isna().all():
        raise AssertionError("All b=5 reference prices must be censored")
    np.testing.assert_allclose(result["within"], [.031078, .021024, .011761, .009747], atol=5.1e-7, rtol=0)
    np.testing.assert_allclose(result["spatial_features"].b1, [.030713,.000033,.029067,.030600,-.000113,.030568,-.001646], atol=5.1e-7, rtol=0)
    np.testing.assert_allclose(result["table7"][["b1","aulc"]].to_numpy(dtype=float),
        [[.222467,.260575],[.253006,.279922],[.223762,.261082],[.254463,.280801],[.254435,.280715],
         [.030538,.019347],[.030673,.019633],[.001429,.000793],[-.000028,-.000086]],atol=5.1e-7,rtol=0)
    loc = result["locations"]
    for budget, regime, feature, value, low, high in [
        (1,"None","Position",.03071,.02796,.03327),
        (1,"Budget-matched","Location",-.000024,-.000042,-.000005),
        (1,"Five repeats","Location",-.000026,-.000047,-.000008),
        (1,"Budget-matched","Position after location",.030721,.027973,.033240),
        (1,"Five repeats","Position after location",.030725,.027987,.033234),
        (2,"Budget-matched","Location",-.000026,-.000038,-.000011),
        (2,"Five repeats","Location",-.000002,-.000051,.000076),
        (2,"Budget-matched","Position after location",.020424,.018248,.022293),
        (2,"Five repeats","Position after location",.020402,.018227,.022309),
    ]:
        row = loc[(loc.budget == budget)&(loc.location_fit == regime)&(loc.added_feature == feature)].iloc[0]
        np.testing.assert_allclose(row[["gain","lower","upper"]].to_numpy(dtype=float), [value, low, high], atol=5.1e-6 if regime=="None" else 5.1e-7, rtol=0)
    if round(result["overstatement_percent"]) != 45:
        raise AssertionError("Headline overstatement percentage differs")
    scan = result["scan"].query("policy_arm == 'selected'").pivot(index=["scan","budget"], columns="candidate", values="mean_target_pearson")
    if len(scan)!=40 or not (scan.D1_DXY > scan.D1_PAIR_CALIBRATED).all():
        raise AssertionError("Expected positive distance gain in all 10 scans and 4 budgets")


def make_figures(result, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    curves = result["curves"]
    styles = {"D1": ("#1f77b4", "o", "--"), "D1_PAIR_CALIBRATED": ("#ff7f0e", "s", "-."),
        "D1_DXY": ("#2ca02c", "D", "-"), "F1": ("#d62728", "^", "-"),
        "A1": ("#9467bd", "v", "-"), "R2": ("#8c564b", "x", ":")}
    def budget_axis(ax):
        ax.set_xscale("log", base=2)
        ax.set_xticks(BUDGETS, [str(b) for b in BUDGETS])
        ax.set_xlabel("Repeats per training movie, b")
    def lines(ax, names):
        for name in names:
            color, marker, style = styles[name]
            ax.plot(BUDGETS, curves[name], label=LABELS[name], color=color, marker=marker, ls=style)
        budget_axis(ax)
        ax.set_ylabel("Mean target Pearson r")
        ax.legend(frameon=False)
    def save(fig, name):
        fig.savefig(out / f"{name}.svg", bbox_inches="tight", metadata={"Creator":"", "Date":None})
        fig.savefig(out / f"{name}.png", dpi=180, bbox_inches="tight", metadata={"Software":""})
        plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(10,3.6), layout="constrained")
    lines(axes[0], ["D1","D1_PAIR_CALIBRATED","D1_DXY"])
    axes[0].set_title("A  Prediction from limited target data",loc="left")
    ref = curves["D1_SHRUNK"]
    grid = np.linspace(1,3.1,200)
    axes[1].plot(grid, np.interp(np.log2(grid), np.log2(BUDGETS), ref), label="Reference curve")
    positions=[]
    for name in ["D1_PAIR_CALIBRATED","D1_DXY"]:
        score=curves[name][0]; x=inverse(score,ref); positions.append(x)
        color,marker,_=styles[name]
        axes[1].plot(x,score,marker=marker,color=color)
        axes[1].annotate(f"{LABELS[name]}\n{x:.3f}",(x,score),xytext=(7,-30),textcoords="offset points",fontsize=9)
    axes[1].add_patch(FancyArrowPatch((positions[0],.200),(positions[1],.200),arrowstyle="|-|",mutation_scale=7))
    axes[1].text(np.mean(positions),.203,f"+{positions[1]-positions[0]:.3f} repeats",ha="center")
    axes[1].set(xlabel="Equivalent repeats on the reference",ylabel="Mean target Pearson r",title="B  One-repeat gain in data units",ylim=(.19,.29))
    save(fig,"figure1_prediction_and_price")
    fig,ax=plt.subplots(figsize=(10,3.4)); ax.set_axis_off()
    boxes=[(.17,.80,"Context training data\nFit map and covariance"),(.71,.80,"Target calibration data\nb repeats of each of four movies"),
           (.15,.46,"Direct\nEmpirical correlations"),(.49,.46,"Mapped\nShared correlation map"),(.83,.46,"Mapped + position\nSame map with soma distance"),
           (.49,.12,"Common covariance decoder + test context activity\nPredict the withheld target response on another movie")]
    for x,y,text in boxes:ax.text(x,y,text,ha="center",va="center",bbox=dict(boxstyle="round,pad=.5",fc="#f3f5f7",ec="#9ba6b2"))
    for x in [.15,.49,.83]:ax.annotate("",xy=(x,.60),xytext=(.50,.71),arrowprops=dict(arrowstyle="->")); ax.annotate("",xy=(.49,.23),xytext=(x,.36),arrowprops=dict(arrowstyle="->"))
    save(fig,"figure2_prediction_design")
    fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout="constrained")
    axes[0].plot(BUDGETS,curves["D1_PAIR_CALIBRATED"]-curves["D1"],"s--",color="#ff7f0e",label="Shared map")
    axes[0].plot(BUDGETS,curves["D1_DXY"]-curves["D1_PAIR_CALIBRATED"],"D-",color="#2ca02c",label="Position beyond map")
    axes[0].axhline(0,color="grey",lw=.6);axes[0].set_ylabel("Change in Pearson r")
    for i,label in [(0,"Log budget"),(1,"Linear budget")]:axes[1].plot(BUDGETS,result["prices"].iloc[i][["b1","b2","b4","b5"]].to_numpy(dtype=float),marker="os"[i],label=label)
    axes[1].text(5,.5,"Censored",rotation=90,ha="center");axes[1].set(ylabel="Position gain in reference repeats",ylim=(0,1.08))
    for ax,title in zip(axes,["A  Gains at a fixed target budget","B  Position gain in reference repeats"]):budget_axis(ax);ax.legend(frameon=False);ax.set_title(title,loc="left")
    save(fig,"figure3_gain_and_equivalent_repeats")
    fig,ax=plt.subplots(figsize=(7.5,4),layout="constrained");lines(ax,list(styles));save(fig,"figure4_profile_learning_curves")
    fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout="constrained")
    rows=result["spatial_features"].iloc[[0,1,2,4,5]]
    labels=["In-plane","Depth","3D","Depth after\nin-plane","In-plane\nafter depth"]
    axes[0].barh(labels,rows.b1);axes[0].invert_yaxis();axes[0].set_xlabel("One-repeat change in r")
    for i,value in enumerate(rows.b1):axes[0].text(max(value,0)+.001,i,f"{value:+.5f}",va="center",fontsize=9)
    axes[0].set_xlim(-.001,.046);axes[0].set_title("A  Spatial feature comparisons",loc="left")
    axes[1].plot(BUDGETS,curves["D1_DXY"]-curves["D1_PAIR_CALIBRATED"],"D-",label="All context")
    axes[1].plot(BUDGETS,result["within"],"s--",label="Same-field context")
    budget_axis(axes[1]);axes[1].set(ylabel="Position gain in Pearson r",ylim=(0,.036),title="B  Restricting context to one field");axes[1].legend(frameon=False)
    save(fig,"figure5_spatial_controls")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=ROOT/"results")
    parser.add_argument("--output",type=Path,default=ROOT/"outputs/paper")
    parser.add_argument("--check",action="store_true",help="Fail if recomputed quantities disagree with manuscript rounding")
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    result=build_tables(args.data,args.output)
    if args.check:check_manuscript(result)
    make_figures(result,args.output)
    receipt={"manuscript_checks":"passed" if args.check else "not requested", "overstatement_percent":result["overstatement_percent"],
             "reproduction_level":"aggregation of retained technical-cell scores; no model refitting",
             "uncertainty":"conditional fixed-prediction movie reweighting, one mouse"}
    (args.output/"checks.json").write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8")
    print(f"Saved manuscript tables and five figures to {args.output}")
    if args.check:print("Manuscript numerical checks passed.")


if __name__=="__main__":main()
