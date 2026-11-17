"""Gym segment descriptions, offer policy, prediction, and Excel/CSV audit outputs.

Run in the provided Colab notebook. The saved k=4 centers are reused without fitting.
"""
import json
import hashlib
from importlib.metadata import version
import math
import shutil
import sys
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

RESULTS_REF = "7c112af30611d319c2f21a0d197e3cf34c010357"
REPOSITORY = "FrankZhao111/ClusterHW"
SELECTED_K = 4
SEGMENT_NAMES = {
    0: "Newer, occasional visitors",
    1: "Long-standing older regulars",
    2: "Frequent visitors",
    3: "Class-oriented regulars",
}
COURSE_ASSUMPTIONS = {
    "spend_threshold": 20.0,
    "credit_cost": 10.0,
    "incremental_spend": 25.0,
    "extras_margin": 0.60,
}
NEW_MEMBER = {
    "age": 30, "visits_per_month": 4, "avg_visit_minutes": 42,
    "classes_per_month": 1, "months_member": 5, "monthly_extras_spend": 10,
}


def github_url(ref, path):
    return f"https://raw.githubusercontent.com/{REPOSITORY}/{ref}/{path}"


def read_json(ref, path):
    with urlopen(github_url(ref, path)) as response:
        return json.load(response)


def make_recommender(model):
    """Return recommend_offer(member_features), a function with one input argument."""
    features = model["features"]
    mean = np.asarray(model["scaling"]["mean"], dtype=float)
    std = np.asarray(model["scaling"]["std"], dtype=float)
    centers = np.asarray(model["centers_scaled"], dtype=float)
    eligible = set(model["eligible_clusters"])
    names = {int(k): v for k, v in model["segment_names"].items()}

    def recommend_offer(member_features):
        """Scale one member, find the nearest saved center, and apply segment policy."""
        missing = [feature for feature in features if feature not in member_features]
        if missing:
            raise ValueError(f"Missing features: {missing}")
        values = np.asarray([float(member_features[f]) for f in features], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("All six member features must be finite numbers.")
        scaled = (values - mean) / std
        distances_sq = np.sum((centers - scaled) ** 2, axis=1)
        cluster = int(np.argmin(distances_sq))
        gets_credit = cluster in eligible
        return {
            "cluster": cluster,
            "segment": names[cluster],
            "gets_credit": gets_credit,
            "credit_amount": model["finance_assumptions"]["credit_cost"] if gets_credit else 0.0,
            "squared_distance": float(distances_sq[cluster]),
            "distance": float(np.sqrt(distances_sq[cluster])),
        }

    return recommend_offer


def run_analysis(output_dir="outputs/business_decision", results_ref=RESULTS_REF):
    """Calculate the required decisions from pinned data and the saved fitted centers."""
    a = COURSE_ASSUMPTIONS.copy()
    checkpoint = read_json(results_ref, "kmeans_outputs/model_checkpoint.json")
    data_ref = checkpoint["data_commit"]
    features = checkpoint["features"]
    mean = np.asarray(checkpoint["scaling"]["mean"], dtype=float)
    std = np.asarray(checkpoint["scaling"]["std"], dtype=float)
    centers = np.asarray(checkpoint["models"][str(SELECTED_K)]["centers_scaled"], dtype=float)
    original_centers = centers * std + mean

    raw = pd.read_csv(github_url(data_ref, "gym_members.csv"))
    saved = pd.read_csv(github_url(results_ref, "kmeans_outputs/candidate_assignments.csv"))
    saved = saved.loc[saved["k"].eq(SELECTED_K),
                      ["member_id", "split", "cluster", "squared_distance"]]
    metrics = pd.read_csv(github_url(results_ref, "kmeans_outputs/fit_metrics.csv"))
    members = raw.merge(saved, on="member_id", validate="one_to_one").sort_values("member_id")
    members = members.reset_index(drop=True)
    assert len(raw) == len(members) == 330
    assert members["member_id"].is_unique
    assert members["split"].value_counts().to_dict() == {"train": 231, "test": 99}
    assert centers.shape == (4, 6)

    # Check that every assignment is still the nearest TRAINING-fitted center.
    X = (members[features].to_numpy(dtype=float) - mean) / std
    all_distances_sq = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    labels = np.argmin(all_distances_sq, axis=1)
    np.testing.assert_array_equal(labels, members["cluster"].to_numpy())
    np.testing.assert_allclose(
        all_distances_sq[np.arange(len(members)), labels],
        members["squared_distance"].to_numpy(), rtol=1e-10, atol=1e-10,
    )

    members["segment"] = members["cluster"].map(SEGMENT_NAMES)
    members["low_spender"] = members["monthly_extras_spend"].lt(a["spend_threshold"]).astype(int)
    members["incremental_revenue_if_credit"] = members["low_spender"] * a["incremental_spend"]
    members["incremental_margin_if_credit"] = (
        members["incremental_revenue_if_credit"] * a["extras_margin"]
    )
    members["baseline_net"] = members["incremental_margin_if_credit"] - a["credit_cost"]

    # Choose eligibility from TRAINING members only. An exact zero is a no-credit tie.
    policy_rows = []
    for cluster in range(SELECTED_K):
        group = members.loc[members["cluster"].eq(cluster)]
        training = group.loc[group["split"].eq("train")]
        train_low = int(training["low_spender"].sum())
        all_low = int(group["low_spender"].sum())
        training_net = float(training["baseline_net"].sum())
        flag = int(training_net > 0)
        reason = ("Positive training contribution" if flag else
                  "Break-even: no credit" if training_net == 0 else
                  "Negative training contribution")
        policy_rows.append({
            "cluster": cluster, "segment": SEGMENT_NAMES[cluster],
            "training_members": len(training),
            "training_low_spenders": train_low,
            "training_high_spenders": len(training) - train_low,
            "training_low_share": train_low / len(training),
            "training_net_if_credit": training_net,
            "gets_credit": flag, "policy_reason": reason,
            "all_members": len(group), "all_low_spenders": all_low,
            "all_high_spenders": len(group) - all_low,
            "all_net_if_credit": float(group["baseline_net"].sum()),
            "v1_net": flag * float(group["baseline_net"].sum()),
        })
    policy = pd.DataFrame(policy_rows)
    flags = policy.set_index("cluster")["gets_credit"]
    members["v1_gets_credit"] = members["cluster"].map(flags).astype(int)
    members["v1_net"] = members["v1_gets_credit"] * members["baseline_net"]
    members["improvement"] = members["v1_net"] - members["baseline_net"]
    members["baseline_credit_cost"] = a["credit_cost"]
    members["v1_credit_cost"] = members["v1_gets_credit"] * a["credit_cost"]
    member_columns = [
        "member_id", "split", "cluster", "segment", *features,
        "low_spender", "incremental_revenue_if_credit", "incremental_margin_if_credit",
        "baseline_net", "v1_gets_credit", "v1_net", "improvement",
        "baseline_credit_cost", "v1_credit_cost", "squared_distance",
    ]
    members = members[member_columns]

    centroid_rows, profile_rows, center_rows = [], [], []
    for cluster in range(SELECTED_K):
        group = members.loc[members["cluster"].eq(cluster)]
        training = group.loc[group["split"].eq("train")]
        np.testing.assert_allclose(
            training[features].mean().to_numpy(), original_centers[cluster],
            rtol=1e-10, atol=1e-10,
        )
        common = {
            "cluster": cluster, "segment": SEGMENT_NAMES[cluster],
            "all_members": len(group), "training_members": len(training),
            "test_members": len(group) - len(training),
            "gets_credit": int(flags.loc[cluster]),
        }
        centroid_rows.append({
            **common, **{f"centroid_{f}": float(original_centers[cluster, j])
                         for j, f in enumerate(features)},
        })
        profile_rows.append({
            **common, **{f"mean_{f}": float(group[f].mean()) for f in features},
        })
        center_rows.append({
            "cluster": cluster, "segment": SEGMENT_NAMES[cluster],
            "training_members": len(training),
            **{f"scaled_{f}": float(centers[cluster, j]) for j, f in enumerate(features)},
            **{f"original_{f}": float(original_centers[cluster, j])
               for j, f in enumerate(features)},
        })
    centroids = pd.DataFrame(centroid_rows)
    profiles = pd.DataFrame(profile_rows)
    centers_table = pd.DataFrame(center_rows)
    scaling = pd.DataFrame({
        "feature": features, "training_mean": mean, "training_std": std,
        "training_count": 231, "ddof": 0,
    })

    comparison_rows = []
    for plan, recipient_flags in [
        ("Baseline: credit to everyone", np.ones(len(members), dtype=int)),
        ("V1: selected segments", members["v1_gets_credit"].to_numpy()),
    ]:
        revenue = float(np.sum(recipient_flags * members["incremental_revenue_if_credit"]))
        margin = revenue * a["extras_margin"]
        cost = float(recipient_flags.sum() * a["credit_cost"])
        comparison_rows.append({
            "plan": plan, "members": len(members), "recipients": int(recipient_flags.sum()),
            "incremental_revenue": revenue, "incremental_margin": margin,
            "credit_cost": cost, "net": margin - cost,
        })
    comparison_rows.append({
        "plan": "Change: V1 minus baseline",
        **{key: comparison_rows[1][key] - comparison_rows[0][key]
           for key in comparison_rows[0] if key != "plan"},
    })
    comparison = pd.DataFrame(comparison_rows)
    by_split = pd.DataFrame([
        {
            "split": split, "members": len(frame),
            "baseline_net": float(frame["baseline_net"].sum()),
            "v1_recipients": int(frame["v1_gets_credit"].sum()),
            "v1_net": float(frame["v1_net"].sum()),
            "improvement": float(frame["improvement"].sum()),
        }
        for split, frame in [
            ("train", members.loc[members["split"].eq("train")]),
            ("test", members.loc[members["split"].eq("test")]), ("all", members),
        ]
    ])
    chosen_metric = metrics.loc[metrics["k"].eq(SELECTED_K)].iloc[0]
    baseline_metric = metrics.loc[metrics["k"].eq(1)].iloc[0]
    fit_summary = pd.DataFrame([
        {
            "split": split, "members": count,
            "baseline_spread": float(baseline_metric[f"{split}_spread"]),
            "selected_k": SELECTED_K,
            "selected_spread": float(chosen_metric[f"{split}_spread"]),
            "reduction_pct": float(chosen_metric[f"{split}_reduction_vs_k1_pct"]),
        }
        for split, count in [("train", 231), ("test", 99)]
    ])

    selected_model = {
        "selected_k": SELECTED_K, "features": features,
        "results_commit": results_ref, "data_commit": data_ref,
        "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scaling": {"mean": mean.tolist(), "std": std.tolist(), "ddof": 0},
        "centers_scaled": centers.tolist(),
        "segment_names": {str(k): v for k, v in SEGMENT_NAMES.items()},
        "eligible_clusters": [int(c) for c in policy.loc[policy["gets_credit"].eq(1), "cluster"]],
        "policy_basis": "Positive training-member incremental contribution; zero means no credit",
        "finance_assumptions": a,
        "fit_configuration": checkpoint["kmeans_configuration"],
        "fit_versions": checkpoint["versions"],
        "analysis_versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                              "pandas": pd.__version__, "xlsxwriter": version("xlsxwriter"),
                              "openpyxl": version("openpyxl")},
    }
    recommend_offer = make_recommender(selected_model)
    new_result = recommend_offer(NEW_MEMBER)
    new_values = np.asarray([NEW_MEMBER[f] for f in features], dtype=float)
    new_z = (new_values - mean) / std
    new_scaling = pd.DataFrame({
        "feature": features, "original_value": new_values,
        "training_mean": mean, "training_std": std, "scaled_value": new_z,
    })
    new_d2 = np.sum((centers - new_z) ** 2, axis=1)
    new_distances = pd.DataFrame({
        "cluster": range(SELECTED_K),
        "segment": [SEGMENT_NAMES[c] for c in range(SELECTED_K)],
        "squared_distance": new_d2, "distance": np.sqrt(new_d2),
        "gets_credit": [int(flags.loc[c]) for c in range(SELECTED_K)],
        "selected": [int(c == new_result["cluster"]) for c in range(SELECTED_K)],
    })
    new_summary = pd.DataFrame([{**NEW_MEMBER, **new_result}])

    # Validate inference against every saved assignment and policy decision.
    for row in members.to_dict("records"):
        prediction = recommend_offer(row)
        assert prediction["cluster"] == row["cluster"]
        assert int(prediction["gets_credit"]) == row["v1_gets_credit"]
    np.testing.assert_allclose(comparison.loc[0, "net"], members["baseline_net"].sum())
    np.testing.assert_allclose(comparison.loc[1, "net"], members["v1_net"].sum())
    np.testing.assert_allclose(comparison.loc[2, "net"], members["improvement"].sum())
    assert centroids["all_members"].sum() == 330
    # Frozen assignment references: these independently established figures catch regressions.
    np.testing.assert_allclose(comparison["net"].to_numpy(), [-1200.0, 185.0, 1385.0])
    assert selected_model["eligible_clusters"] == [0]
    assert new_result["cluster"] == 0 and new_result["gets_credit"]

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tables = {
        "segment_policy": policy, "segment_summary": centroids,
        "segment_population_profile": profiles, "member_audit": members,
        "plan_comparison": comparison, "value_by_split": by_split,
        "model_fit_summary": fit_summary, "fit_metrics": metrics,
        "scaling_parameters": scaling, "selected_centers": centers_table,
        "new_member_summary": new_summary, "new_member_scaling": new_scaling,
        "new_member_distances": new_distances,
    }
    for name, frame in tables.items():
        frame.to_csv(output / f"{name}.csv", index=False, float_format="%.17g")
    (output / "selected_model.json").write_text(json.dumps(selected_model, indent=2) + "\n",
                                               encoding="utf-8")
    result = {
        "tables": tables, "selected_model": selected_model, "new_member": new_result,
        "output_dir": output, "recommend_offer": recommend_offer,
    }
    workbook_path = output / "business_analysis.xlsx"
    write_audit_workbook(workbook_path, result)
    validate_workbook_cache(workbook_path, result)
    result["workbook_path"] = workbook_path
    result["archive_path"] = shutil.make_archive("business_outputs", "zip", root_dir=output)
    return result


def write_audit_workbook(path, result):
    """Write Excel formulas with Python-calculated result caches for preview applications."""
    import xlsxwriter
    from xlsxwriter.utility import xl_col_to_name

    t, model = result["tables"], result["selected_model"]
    features, a = model["features"], model["finance_assumptions"]
    policy, members = t["segment_policy"], t["member_audit"]
    new = result["new_member"]
    assumptions = pd.DataFrame([
        ["spend_threshold", a["spend_threshold"], "Below this spend level the credit generates incremental spending."],
        ["credit_cost", a["credit_cost"], "Every recipient redeems this amount; cost per member per month."],
        ["incremental_spend", a["incremental_spend"], "Extra spending by a below-threshold recipient."],
        ["extras_margin", a["extras_margin"], "Contribution margin on the incremental spending."],
    ], columns=["assumption", "value", "meaning"])
    guide = pd.DataFrame([
        ["Start here", "Read Model_Fit, Segment_Summary, Segment_Policy, Plan_Comparison, and New_Member."],
        ["Professor's format", "Python code and analysis are in GitHub. Submit one PDF: title page plus two student-written content slides."],
        ["Model", "Four saved training-fitted centers; six standardized features. Member ID is excluded from distances."],
        ["Split", "231 training members and 99 test members; all 330 enter the required policy simulation."],
        ["Policy", "Credit a segment only if its training-member contribution is positive. An exact zero is a no-credit tie."],
        ["Member economics", "Below 20: 25 x 60% - 10 = +5. At or above 20: no incremental margin and a 10 cost, giving -10."],
        ["Contribution", "Financial values are incremental monthly contribution from the credit, not total gym profit."],
        ["Centroids", "Segment_Summary uses training centroids. Full_Profiles uses averages of all assigned members."],
        ["Source records", "Members and Centers are fixed records from this model run; changing historical features requires rerunning Python."],
        ["Input cells", "Blue cells on Assumptions and New_Member are editable what-if inputs. CSVs retain the original notebook-run values."],
        ["Formula audit", "Financials, policy, plan totals, scaling, original-unit centers, new-member distances, and fit reductions contain Excel formulas."],
        ["Formula previews", "Python supplies formula result caches. Excel recalculates formulas when opened."],
        ["New member", "The supplied example is outside the 330 historical members and excluded from plan totals."],
        ["Labels", "Names are working descriptions. Cluster numbers are identifiers, not rankings."],
        ["Financial evidence", "The figures follow Finance's supplied assumptions; they are a simulation, not measured campaign effects."],
        ["Data commit", model["data_commit"]],
        ["Candidate results commit", model["results_commit"]],
    ], columns=["item", "explanation"])
    sheets = {
        "Guide": guide, "Assumptions": assumptions, "Scaling": t["scaling_parameters"],
        "Centers": t["selected_centers"], "Members": members,
        "Segment_Policy": policy, "Segment_Summary": t["segment_summary"],
        "Full_Profiles": t["segment_population_profile"],
        "Plan_Comparison": t["plan_comparison"], "By_Split": t["value_by_split"],
        "Model_Fit": t["model_fit_summary"], "Fit_Comparison": t["fit_metrics"],
        "New_Member": t["new_member_scaling"],
    }
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        wb = writer.book
        wb.set_calc_mode("auto")
        header = wb.add_format({"bold": True, "bg_color": "#24476B", "font_color": "white",
                                "text_wrap": True, "valign": "top", "border": 1})
        number = wb.add_format({"num_format": "0.0000"})
        integer = wb.add_format({"num_format": "0"})
        money = wb.add_format({"num_format": "$#,##0.00;[Red]($#,##0.00)"})
        percent = wb.add_format({"num_format": "0.0%"})
        pct_points = wb.add_format({"num_format": '0.00"%"'})
        text = wb.add_format({"text_wrap": True, "valign": "top"})
        input_number = wb.add_format({"font_color": "#0070C0", "bg_color": "#EAF2F8", "num_format": "0.00"})
        input_percent = wb.add_format({"font_color": "#0070C0", "bg_color": "#EAF2F8", "num_format": "0.0%"})
        highlight = wb.add_format({"bg_color": "#E2F0D9"})

        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
            ws = writer.sheets[name]
            ws.freeze_panes(1, 2 if name == "Members" else 1)
            ws.set_zoom(90)
            ws.set_row(0, 42)
            ws.set_column(0, len(frame.columns) - 1, 19, number)
            ws.set_column(0, 0, 15)
            for j, column in enumerate(frame.columns):
                ws.write(0, j, column, header)
            if name not in {"Guide", "Assumptions", "New_Member"}:
                ws.autofilter(0, 0, len(frame), len(frame.columns) - 1)

        wsmap = writer.sheets
        end = len(members) + 1
        mr = lambda col: "'Members'!$" + col + "$2:$" + col + "$" + str(end)
        pr = lambda col: "'Segment_Policy'!$" + col + "$2:$" + col + "$5"

        def formula(sheet, row, col, expression, cached, fmt=number):
            if isinstance(cached, np.generic):
                cached = cached.item()
            wsmap[sheet].write_formula(row, col, expression, fmt, cached)

        wsmap["Guide"].set_column(0, 0, 27)
        wsmap["Guide"].set_column(1, 1, 105, text)
        for row in range(1, len(guide) + 1):
            wsmap["Guide"].set_row(row, 34)
        wsmap["Assumptions"].set_column(0, 0, 25)
        wsmap["Assumptions"].set_column(2, 2, 85, text)
        for i, row in assumptions.iterrows():
            wsmap["Assumptions"].write(i + 1, 1, float(row["value"]),
                                       input_percent if i == 3 else input_number)
        for row_index, label, value in [(6, "Selected k (fixed model)", SELECTED_K),
                                        (7, "Training members", 231),
                                        (8, "Test members", 99), (9, "All members", 330)]:
            wsmap["Assumptions"].write(row_index, 0, label)
            wsmap["Assumptions"].write_number(row_index, 1, value, integer)
        formula("Assumptions", 7, 1, f'=COUNTIF({mr("B")},"train")', 231, integer)
        formula("Assumptions", 8, 1, f'=COUNTIF({mr("B")},"test")', 99, integer)
        formula("Assumptions", 9, 1, f'=COUNTA({mr("A")})', 330, integer)

        # Members A:J are source values. K:S show the financial calculation.
        wsmap["Members"].set_column(0, 0, 12, integer)
        wsmap["Members"].set_column(1, 2, 12)
        wsmap["Members"].set_column(3, 3, 35, text)
        for i, row in members.iterrows():
            r, er = i + 1, i + 2
            match = f'MATCH(C{er},{pr("A")},0)'
            formula("Members", r, 3, f'=INDEX({pr("B")},{match})', row["segment"], text)
            expressions = {
                10: (f'=IF(J{er}<Assumptions!$B$2,1,0)', row["low_spender"], integer),
                11: (f'=K{er}*Assumptions!$B$4', row["incremental_revenue_if_credit"], money),
                12: (f'=L{er}*Assumptions!$B$5', row["incremental_margin_if_credit"], money),
                13: (f'=M{er}-Assumptions!$B$3', row["baseline_net"], money),
                14: (f'=INDEX({pr("H")},{match})', row["v1_gets_credit"], integer),
                15: (f'=O{er}*N{er}', row["v1_net"], money),
                16: (f'=P{er}-N{er}', row["improvement"], money),
                17: ('=Assumptions!$B$3', row["baseline_credit_cost"], money),
                18: (f'=O{er}*Assumptions!$B$3', row["v1_credit_cost"], money),
            }
            for col, (expression, value, fmt) in expressions.items():
                formula("Members", r, col, expression, value, fmt)

        for j, row in t["scaling_parameters"].iterrows():
            r, er, raw_col = j + 1, j + 2, xl_col_to_name(4 + j)
            formula("Scaling", r, 1,
                    f'=AVERAGEIF({mr("B")},"train",{mr(raw_col)})', row["training_mean"])
            formula("Scaling", r, 2,
                    f'=SQRT(SUMPRODUCT(({mr("B")}="train")*({mr(raw_col)}-B{er})^2)/D{er})',
                    row["training_std"])
            formula("Scaling", r, 3, f'=COUNTIF({mr("B")},"train")', 231, integer)

        # Segment_Policy H depends on G, which sums Members N, not the V1 flag.
        wsmap["Segment_Policy"].set_column(1, 1, 35, text)
        wsmap["Segment_Policy"].set_column(8, 8, 34, text)
        for i, row in policy.iterrows():
            r, er = i + 1, i + 2
            expressions = {
                2: (f'=COUNTIFS({mr("C")},A{er},{mr("B")},"train")', row["training_members"], integer),
                3: (f'=SUMIFS({mr("K")},{mr("C")},A{er},{mr("B")},"train")', row["training_low_spenders"], integer),
                4: (f'=C{er}-D{er}', row["training_high_spenders"], integer),
                5: (f'=D{er}/C{er}', row["training_low_share"], percent),
                6: (f'=SUMIFS({mr("N")},{mr("C")},A{er},{mr("B")},"train")', row["training_net_if_credit"], money),
                7: (f'=IF(G{er}>0,1,0)', row["gets_credit"], integer),
                8: (f'=IF(G{er}>0,"Positive training contribution",IF(G{er}=0,"Break-even: no credit","Negative training contribution"))',
                    row["policy_reason"], text),
                9: (f'=COUNTIF({mr("C")},A{er})', row["all_members"], integer),
                10: (f'=SUMIF({mr("C")},A{er},{mr("K")})', row["all_low_spenders"], integer),
                11: (f'=J{er}-K{er}', row["all_high_spenders"], integer),
                12: (f'=SUMIF({mr("C")},A{er},{mr("N")})', row["all_net_if_credit"], money),
                13: (f'=H{er}*M{er}', row["v1_net"], money),
            }
            for col, (expression, value, fmt) in expressions.items():
                formula("Segment_Policy", r, col, expression, value, fmt)

        wsmap["Centers"].set_column(1, 1, 35, text)
        for i, row in t["selected_centers"].iterrows():
            r, er = i + 1, i + 2
            formula("Centers", r, 2,
                    f'=COUNTIFS({mr("C")},A{er},{mr("B")},"train")', row["training_members"], integer)
            for j, feature in enumerate(features):
                col = xl_col_to_name(3 + j)
                std_ref = "Scaling!$C$" + str(j + 2)
                mean_ref = "Scaling!$B$" + str(j + 2)
                formula("Centers", r, 9 + j,
                        f'={col}{er}*{std_ref}+{mean_ref}', row[f"original_{feature}"])

        for sheet_name, table_name, prefix in [
            ("Segment_Summary", "segment_summary", "centroid_"),
            ("Full_Profiles", "segment_population_profile", "mean_"),
        ]:
            wsmap[sheet_name].set_column(1, 1, 35, text)
            for i, row in t[table_name].iterrows():
                r, er = i + 1, i + 2
                for col, expression, value in [
                    (2, f"='Segment_Policy'!J{er}", row["all_members"]),
                    (3, f"='Segment_Policy'!C{er}", row["training_members"]),
                    (4, f'=C{er}-D{er}', row["test_members"]),
                    (5, f"='Segment_Policy'!H{er}", row["gets_credit"]),
                ]:
                    formula(sheet_name, r, col, expression, value, integer)
                for j, feature in enumerate(features):
                    expression = (f"=Centers!{xl_col_to_name(9 + j)}{er}" if prefix == "centroid_"
                                  else f'=AVERAGEIF({mr("C")},A{er},{mr(xl_col_to_name(4 + j))})')
                    formula(sheet_name, r, 6 + j, expression, row[prefix + feature])

        pc = t["plan_comparison"]
        wsmap["Plan_Comparison"].set_column(0, 0, 34, text)
        for i in [0, 1]:
            r, er = i + 1, i + 2
            row = pc.iloc[i]
            formula("Plan_Comparison", r, 1, f'=COUNTA({mr("A")})', row["members"], integer)
            expressions = (
                [f'=COUNTA({mr("A")})', f'=SUM({mr("L")})', f'=SUM({mr("M")})', f'=SUM({mr("R")})']
                if i == 0 else
                [f'=SUM({mr("O")})', f'=SUMPRODUCT({mr("O")},{mr("L")})',
                 f'=SUMPRODUCT({mr("O")},{mr("M")})', f'=SUM({mr("S")})']
            )
            for col, expression, key in zip(range(2, 6), expressions,
                                            ["recipients", "incremental_revenue", "incremental_margin", "credit_cost"]):
                formula("Plan_Comparison", r, col, expression, row[key], integer if col == 2 else money)
            formula("Plan_Comparison", r, 6, f'=E{er}-F{er}', row["net"], money)
        for col in range(1, 7):
            letter = xl_col_to_name(col)
            formula("Plan_Comparison", 3, col, f'={letter}3-{letter}2',
                    pc.iloc[2, col], integer if col < 3 else money)
        wsmap["Plan_Comparison"].write(0, 7, "Member-total check", header)
        formula("Plan_Comparison", 1, 7, f'=G2-SUM({mr("N")})', 0.0, money)
        formula("Plan_Comparison", 2, 7, f'=G3-SUM({mr("P")})', 0.0, money)

        for i, row in t["value_by_split"].iterrows():
            r, er = i + 1, i + 2
            all_rows = row["split"] == "all"
            count_expr = f'=COUNTA({mr("A")})' if all_rows else f'=COUNTIF({mr("B")},A{er})'
            formula("By_Split", r, 1, count_expr, row["members"], integer)
            for col, member_col, key in [
                (2, "N", "baseline_net"), (3, "O", "v1_recipients"), (4, "P", "v1_net")
            ]:
                expression = (f'=SUM({mr(member_col)})' if all_rows else
                              f'=SUMIF({mr("B")},A{er},{mr(member_col)})')
                formula("By_Split", r, col, expression, row[key], integer if col == 3 else money)
            formula("By_Split", r, 5, f'=E{er}-C{er}', row["improvement"], money)

        for i, row in t["fit_metrics"].iterrows():
            r, er = i + 1, i + 2
            formula("Fit_Comparison", r, 3, f'=B{er}/Assumptions!$B$8', row["train_spread_per_member"])
            formula("Fit_Comparison", r, 4, f'=C{er}/Assumptions!$B$9', row["test_spread_per_member"])
            if i > 0:
                formula("Fit_Comparison", r, 6, f'=B{er - 1}-B{er}', row["train_drop_from_previous"])
                formula("Fit_Comparison", r, 7, f'=C{er - 1}-C{er}', row["test_drop_from_previous"])
            formula("Fit_Comparison", r, 8, f'=100*(1-B{er}/$B$2)', row["train_reduction_vs_k1_pct"], pct_points)
            formula("Fit_Comparison", r, 9, f'=100*(1-C{er}/$C$2)', row["test_reduction_vs_k1_pct"], pct_points)
        wsmap["Fit_Comparison"].conditional_format(4, 0, 4, 9, {"type": "no_blanks", "format": highlight})
        for i, row in t["model_fit_summary"].iterrows():
            r, er, spread_col = i + 1, i + 2, "B" if row["split"] == "train" else "C"
            source_range = "'Fit_Comparison'!$" + spread_col + "$2:$" + spread_col + "$9"
            formula("Model_Fit", r, 1, f'=COUNTIF({mr("B")},A{er})', row["members"], integer)
            formula("Model_Fit", r, 2, f"='Fit_Comparison'!{spread_col}2", row["baseline_spread"])
            formula("Model_Fit", r, 3, '=Assumptions!$B$7', SELECTED_K, integer)
            formula("Model_Fit", r, 4,
                    f'=INDEX({source_range},MATCH(D{er},Fit_Comparison!$A$2:$A$9,0))',
                    row["selected_spread"])
            formula("Model_Fit", r, 5, f'=100*(1-E{er}/C{er})', row["reduction_pct"], pct_points)

        # Native Excel charts use the same eight model-spread rows as the earlier plots.
        for spread_col, title, position, color in [
            (1, "Training spread (231 members)", "A12", "#245A81"),
            (2, "Test spread (99 members)", "J12", "#B45F06"),
        ]:
            chart = wb.add_chart({"type": "line"})
            chart.add_series({"name": title, "categories": ["Fit_Comparison", 1, 0, 8, 0],
                              "values": ["Fit_Comparison", 1, spread_col, 8, spread_col],
                              "line": {"color": color}, "marker": {"type": "circle", "size": 5}})
            chart.set_title({"name": title})
            chart.set_x_axis({"name": "Number of clusters (k)"})
            chart.set_y_axis({"name": "Sum of squared distances", "min": 0})
            chart.set_legend({"none": True})
            chart.set_size({"width": 570, "height": 340})
            wsmap["Fit_Comparison"].insert_chart(position, chart)

        ws = wsmap["New_Member"]
        ws.set_column(0, 0, 27)
        ws.set_column(1, 1, 35, text)
        for j, row in t["new_member_scaling"].iterrows():
            r, er = j + 1, j + 2
            ws.write_number(r, 1, float(row["original_value"]), input_number)
            formula("New_Member", r, 2, f'=Scaling!B{er}', row["training_mean"])
            formula("New_Member", r, 3, f'=Scaling!C{er}', row["training_std"])
            formula("New_Member", r, 4, f'=(B{er}-C{er})/D{er}', row["scaled_value"])
        for j, label in enumerate(t["new_member_distances"].columns):
            ws.write(10, j, label, header)
        for i, row in t["new_member_distances"].iterrows():
            r, er, center_er = i + 11, i + 12, i + 2
            ws.write_number(r, 0, int(row["cluster"]), integer)
            match = f'MATCH(A{er},{pr("A")},0)'
            formula("New_Member", r, 1, f'=INDEX({pr("B")},{match})', row["segment"], text)
            terms = [
                "($E$" + str(j + 2) + "-Centers!$" + xl_col_to_name(3 + j) +
                "$" + str(center_er) + ")^2" for j in range(len(features))
            ]
            formula("New_Member", r, 2, "=" + "+".join(terms), row["squared_distance"])
            formula("New_Member", r, 3, f'=SQRT(C{er})', row["distance"])
            formula("New_Member", r, 4, f'=INDEX({pr("H")},{match})', row["gets_credit"], integer)
            formula("New_Member", r, 5, f'=IF(A{er}=$B$18,1,0)', row["selected"], integer)
        summary = [
            (17, "Assigned cluster", '=INDEX($A$12:$A$15,MATCH(MIN($C$12:$C$15),$C$12:$C$15,0))', new["cluster"], integer),
            (18, "Segment", '=INDEX($B$12:$B$15,MATCH($B$18,$A$12:$A$15,0))', new["segment"], text),
            (19, "Gets credit (1=yes, 0=no)", '=INDEX($E$12:$E$15,MATCH($B$18,$A$12:$A$15,0))', int(new["gets_credit"]), integer),
            (20, "Credit amount", '=B20*Assumptions!$B$3', new["credit_amount"], money),
        ]
        for r, label, expression, value, fmt in summary:
            ws.write(r, 0, label, text)
            formula("New_Member", r, 1, expression, value, fmt)
        formula("New_Member", 19, 2, '=IF(B20=1,"Yes","No")', "Yes" if new["gets_credit"] else "No", text)
        ws.write(23, 0, "This example is excluded from the 330-member plan totals.", text)
        ws.set_row(23, 38)
        ws.conditional_format(11, 0, 14, 5,
                              {"type": "formula", "criteria": "=$F12=1", "format": highlight})


def validate_workbook_cache(path, result):
    """Validate cached outcomes and formula presence; Excel recalculates on open."""
    from openpyxl import load_workbook
    cached = load_workbook(path, data_only=True, read_only=True)
    checks = [
        ("Plan_Comparison", "G2", float(result["tables"]["plan_comparison"].iloc[0]["net"])),
        ("Plan_Comparison", "G3", float(result["tables"]["plan_comparison"].iloc[1]["net"])),
        ("Plan_Comparison", "G4", float(result["tables"]["plan_comparison"].iloc[2]["net"])),
        ("Plan_Comparison", "H2", 0.0), ("Plan_Comparison", "H3", 0.0),
        ("New_Member", "B18", result["new_member"]["cluster"]),
        ("New_Member", "B19", result["new_member"]["segment"]),
        ("New_Member", "B20", int(result["new_member"]["gets_credit"])),
        ("New_Member", "B21", result["new_member"]["credit_amount"]),
    ]
    checks.extend(("Segment_Policy", f"H{i + 2}", int(row["gets_credit"]))
                  for i, row in result["tables"]["segment_policy"].iterrows())
    for sheet, cell, expected in checks:
        actual = cached[sheet][cell].value
        if isinstance(expected, str):
            assert actual == expected, (sheet, cell, actual, expected)
        else:
            np.testing.assert_allclose(float(actual), expected, rtol=1e-10, atol=1e-10)
    cached.close()

    formulas = load_workbook(path, data_only=False, read_only=True)
    for sheet, cell, _ in checks:
        assert formulas[sheet][cell].data_type == "f", (sheet, cell, "missing formula")
    formula_count = sum(
        cell.data_type == "f" for ws in formulas.worksheets
        for row in ws.iter_rows() for cell in row
    )
    formulas.close()
    assert formula_count > 330 * 5
    result["validation"] = {
        "cached_results_checked": len(checks), "excel_formula_cells": formula_count,
        "status": "Python/data checks and workbook cache/structure checks passed",
    }


if __name__ == "__main__":
    result = run_analysis()
    print(result["tables"]["plan_comparison"].to_string(index=False))
    print("New member:", result["new_member"])
    print("Workbook:", result["workbook_path"])
    print("Archive:", result["archive_path"])
