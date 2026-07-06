import ultraplot as pro


def setup_plots():
    pro.rc["figure.dpi"] = 300
    pro.rc["cycle"] = "ggplot"
    pro.rc["image.origin"] = "lower"
    pro.rc["cmap"] = "magma"
    pro.rc["colorbar.width"] = 0.1
    pro.rc["figure.max_open_warning"] = False


def save_figure(figure, path):
    figure.savefig(str(path) + ".pdf")
    figure.savefig(str(path) + ".png")
