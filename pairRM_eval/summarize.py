import json
import glob
import numpy as np
import argparse

ranking_pattern = "data/ranking/test*.jsonl"
#save_path = "figs/heatmap.png"

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        type=str,
        default="data",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    save_path = f'figs/heatmap_{args.name}.png'
    ranking_files = glob.glob(ranking_pattern)
    rankings = []
    for ranking_file in ranking_files:
        with open(ranking_file, "r") as f:
            for line in f:
                rankings.append(json.loads(line))
    print(len(rankings))
    model_names = rankings[0]["model_names"]

    model_win_rate = {
        model_name: {model_name_2: 0 for model_name_2 in model_names}
        for model_name in model_names
    }

    for ranking in rankings:
        for i, model_name in enumerate(model_names):
            for j, model_name_2 in enumerate(model_names):
                if i == j:
                    continue
                # assert ranking["ranks"][i][j] == -ranking["ranks"][j][i]
                # if ranking["ranks"][i][j] > 0:
                #     model_win_rate[model_name][model_name_2] += 1
                # elif ranking["ranks"][i][j] == 0:
                #     model_win_rate[model_name][model_name_2] += 0.5
                # if ranking['ranks'][i] > ranking['ranks'][j]:
                #     model_win_rate[model_name][model_name_2] += 1
                # elif ranking['ranks'][i] == ranking['ranks'][j]:
                #     model_win_rate[model_name][model_name_2] += 0.5
                model_win_rate[model_name][model_name_2] += 1/(1+np.exp(ranking['ranks'][j]-ranking['ranks'][i]))

    print(model_win_rate)
    # import pdb
    # pdb.set_trace()
    average_wins = []

    for model_name in model_names:
        average_win = 0
        for model_name_2 in model_names:
            if model_name == model_name_2:
                continue

            average_win += model_win_rate[model_name][model_name_2]

            assert int(np.round(model_win_rate[model_name][model_name_2] + model_win_rate[model_name_2][model_name])) == len(rankings), f"{model_name} vs {model_name_2}: {model_win_rate[model_name][model_name_2]} + {model_win_rate[model_name_2][model_name]} != {len(rankings)}"

        average_win /= len(model_names) - 1
        average_wins.append((model_name, average_win))
        print(f"{model_name}: {average_win}")

    # rank model_names based on average_win
    model_names = [x[0] for x in sorted(average_wins, key=lambda x: -x[1])]

    # plot a heatmap

    import matplotlib.pyplot as plt
    import numpy as np

    win_matrix = np.zeros((len(model_names), len(model_names))) + 0.5

    for i, model_name in enumerate(model_names):
        for j, model_name_2 in enumerate(model_names):
            if i == j:
                continue
            else:
                win_matrix[i][j] = model_win_rate[model_name][model_name_2] / len(
                    rankings
                )

    import plotly.express as px

    # valid_names = [
    #     1 if "UCLA" in name or "Mistral" in name else 0 for name in model_names
    # ]

    # model_names = [name for name, valid in zip(model_names, valid_names) if valid]

    # valid_names = np.array(valid_names) > 0
    # win_matrix = win_matrix[valid_names][:, valid_names]

    def visualize_pairwise_win_fraction(battles, model_order, scale=1):
        row_beats_col = battles
        fig = px.imshow(
            row_beats_col,
            color_continuous_scale="RdBu",
            text_auto=".3f",
            height=700 * scale,
            width=700 * scale,
        )
        fig.update_layout(
            xaxis_title="Model B",
            yaxis_title="Model A",
            xaxis_side="top",
            title_y=0.07,
            title_x=0.5,
        )
        fig.update_traces(
            hovertemplate="Model A: %{y}<br>Model B: %{x}<br>Fraction of A Wins: %{z}<extra></extra>"
        )

        model_order = [_.split("test_")[-1] for _ in model_order]
        model_order = [_.split("train_")[-1] for _ in model_order]
        model_order = [
            (
                _.split("-", 1)[0] + "/" + _.split("-", 1)[1]
                if "-" in _
                else "Snorkel-" + _
            )
            for _ in model_order
        ]

        # add model names
        if model_order is not None:
            fig.update_xaxes(
                ticktext=model_order, tickvals=list(range(len(model_order)))
            )
            fig.update_yaxes(
                ticktext=model_order, tickvals=list(range(len(model_order)))
            )

        # mask out the diagonal
        for i in range(len(model_order)):
            fig.add_shape(
                type="rect",
                x0=i - 0.5,
                x1=i + 0.5,
                y0=i - 0.5,
                y1=i + 0.5,
                line=dict(color="black", width=1),
                fillcolor="white",
            )

        return fig

    fig = visualize_pairwise_win_fraction(win_matrix, model_names, scale=10.0)
    # fig.show()
    fig.write_image(save_path)
