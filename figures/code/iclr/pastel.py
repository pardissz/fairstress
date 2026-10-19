import matplotlib as mpl
LIGHT = dict(blue='#77AADD', cyan='#99DDFF', mint='#44BB99', pear='#BBCC33', olive='#AAAA00', yellow='#EEDD88', orange='#EE8866', pink='#FFAABB', grey='#DDDDDD')
INK = dict(rose='#CC6677', blue='#4477AA', teal='#44AA99', indigo='#332288', sand='#C9A94B', wine='#882255', grey='#777777', orange='#D9784E')
MINORITY = INK['rose']
MAJORITY = INK['blue']

def style():
    mpl.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Nimbus Roman', 'STIXGeneral', 'DejaVu Serif'], 'mathtext.fontset': 'stix', 'font.size': 8.5, 'axes.titlesize': 9, 'axes.labelsize': 8.5, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5, 'axes.edgecolor': '#888888', 'axes.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': 'white', 'axes.facecolor': 'white', 'legend.frameon': False, 'pdf.fonttype': 42})
