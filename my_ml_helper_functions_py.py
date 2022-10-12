# %% [code] {"execution":{"iopub.status.busy":"2022-08-09T21:52:22.113665Z","iopub.execute_input":"2022-08-09T21:52:22.114150Z","iopub.status.idle":"2022-08-09T21:52:22.420650Z","shell.execute_reply.started":"2022-08-09T21:52:22.114112Z","shell.execute_reply":"2022-08-09T21:52:22.419058Z"}}


# Helper functions (general)


import gc
import time
import sys

from IPython.display import Audio, display

import numpy as np
import pandas as pd
import sklearn

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from contextlib import contextmanager




print('\nLibraries version :\n')
print('Python     : ' + sys.version)
print('NumPy      : ' + np.__version__)
print('Pandas     : ' + pd.__version__)
print('Matplotlib : ' + mpl.__version__)
print('Seaborn    : ' + sns.__version__)
print('SkLearn    : ' + sklearn.__version__)

"""
On Kaggle:
Libraries version :
Python     : 3.7.12 | packaged by conda-forge
NumPy      : 1.19.5 now 1.21.5 (april'22)
Pandas     : 1.3.4 now 1.3.5
Matplotlib : 3.5.1
Seaborn    : 0.11.2
SKlearn    : 0.23.2  #1.02 since april'22

On Kaggle (with latest environment option):
Libraries version :
Python     : 3.7.12 | packaged by conda-forge
NumPy      : 1.20.3
Pandas     : 1.3.5
Matplotlib : 3.5.1
Seaborn    : 0.11.2
SKlearn    : 1.0.1
"""

# # ipympl have pb installation in kaggle
# # %conda install ipymp
# # %matplotlib widget

# # %matplotlib qt
# # %matplotlib notebook
# # to get interactive plot (can zoom & horiz. scroll but change layout
# #   from inline)   # or %matplotlib qt ?
# # usage error when comments after jupyter magic command
# %matplotlib inline

# #custom_params = {"axes.spines.right": False, "axes.spines.top": False}
# sns.set_theme(context='paper')   #rc=custom_params
# # context: "notebook"(default), “paper”, “talk”, and “poster”
# #   for scaling/sizes
# # style: darkgrid(default), whitegrid, dark, white, ticks  #for general
# #   style of the plots
# # palette: deep(default), muted, bright, pastel, dark, colorblind, ...
# #   paste1,..., set2, ...,
# #   ‘light:<color>’, ‘dark:<color>’, ‘blend:<color>,<color>
# #   https://seaborn.pydata.org/tutorial/color_palettes.html


# # pd.set_option('display.width', 120)
# pd.set_option('display.max_columns', None)  # print all columns
# # pd.set_option('display.show_dimensions', True)  # to always show the
# #   dimensions of the df when printed

# # # np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
# np.set_printoptions(edgeitems=5, linewidth=120,
#                     formatter={'float': '{: 0.3f}'.format})


nl = "\n"

CEND    = '\33[0m'  ; CBOLD   = '\33[1m'  ; CITALIC = '\33[3m'
CUDL    = '\33[4m'  ; CYELLOW = '\33[33m' ; CWHITE  = '\33[37m'
CBLACK  = '\33[30m' ; CRED    = '\33[31m' ; CGREEN  = '\33[32m'
CBLUE   = '\33[34m' ; CVIOLET = '\33[35m' ; CBEIGE  = '\33[36m'




def myprint(text, color, bold=False, udl=False, italic=False):
    """Print in color, bold, underlined or italic."""
    # [Or use html from ipython]

    s = ''
    if bold:
        s += CBOLD
    if udl:
        s += CUDL
    if italic:
        s += CITALIC
    s += color + text + CEND + '\n'
    print(s)



def explore(df, miss=True, valcount=True):
    """Do basic exploration on a dataframe."""
    # pb when lists in the column and do df.count_values()
    #  the pb is the memory size, it works for 10k lines but not 100k

    myprint('\n----- FIRST LINES -----', CVIOLET, bold=True, udl=True,
            italic=False)
    display(df.head(5))

    myprint('\n----- INFO -----', CVIOLET, bold=True, udl=True,
            italic=False)
    display(pd.DataFrame(df.info(memory_usage='deep', verbose=True,
                                 show_counts=True)))

    myprint('\n----- DESCRIBE -----', CVIOLET, bold=True, udl=True,
            italic=False)
    try:
        display(pd.DataFrame(df.describe(include=['number'])))
        display(pd.DataFrame(df.describe(include=['object'])))
        # display(pd.DataFrame(df.describe(include=['category'])))
    except ValueError:  # if no number columns or no object columns
        pass

    # Unique and missing values per column
    if miss:
        uni_miss(df)

    # Value counts per column
    if valcount:
        myvalcount(df, 10)



def uni_miss(df):
    """
    For the chosen columns of a DF, gives the number/% of missing values.
    and the number of unique values
    Example: uni_miss(data0[['len code']])
    """
    print('\n-----', namestr(df, globals()))  # [0])
    myprint('----- MISSING AND UNIQUE VALUES -----', CVIOLET, bold=True,
            udl=True, italic=False)

    df1 = pd.Series(df.isna().mean())
    df2 = pd.Series(df.isna().sum())
    df3 = pd.concat([df1, df2], axis=1)
    df3.columns = ['%missing', '#missing']
    df3['#values'] = df.shape[0] - df3['#missing']
    if isinstance(df, pd.DataFrame):
        df3['#unique'] = [len(df[c].apply(lambda x: str(x)).unique())
                          for c in df.columns]  # includes NaN
        # tod: substract 1 when NaN in list to have count w/o NaN
        df3['type'] = [df[c].dtypes for c in df.columns]
    else:   # if df is a Series or one column of a DataFrame
        df3['#unique'] = [len(df.apply(lambda x: str(x)).unique())]
            # includes NaN
        df3['type'] = [df.dtypes]

    # Use incremental number index to get number of indicators
    df3['column'] = df3.index
    df3.reset_index(drop=True, inplace=True)
    df3.sort_values('%missing', inplace=True)
    df3.index = np.arange(1, len(df3) + 1)

    # Arrange the order of the columns
    cols = list(df3.columns)
    cols = [cols[-1]] + cols[:-1]
    df3 = df3[cols]

    with pd.option_context('display.max_rows', None):
        display(df3.sort_values(by='%missing'))



def myvalcount(df, n_top=10):
    """
    For each DF column, gives the list of the n_top most frequent values.

    and the number of occurences
    Example: myvalcount(data0[['len code']])
    Too slow (jam) if 100k lines of lists
    """
    if isinstance(df, pd.DataFrame):
        cols = df.columns
    else:   # if df is a Series or one column of a DataFrame
        cols = df.name
    myprint('\n----- VALUE COUNTS -----', CVIOLET, bold=True, udl=True,
            italic=False)
    for col in cols:
        res = pd.DataFrame(df[col].value_counts(dropna=False)).head(n_top)
        myprint('\nTop ' + str(n_top) + ' value counts for ' + str(col) +
                ' (' + str(len(res)) + ' mod.):',
                CGREEN, bold=True, udl=True, italic=False)
        display(res)
    return res



def visudf(df, title='', n=10):
    """Display some infos on a dataframe/series."""
    # print('\nName:', namestr(df, locals())[0])
    print('\n', title, ':')
    print('Object:', type(df))
    print('Shape:', df.shape)
    print('Types:', df.dtypes)
    mydisplay(df.head(n))
    print('\n')



def mem_usage(pandas_obj, list_typ):
    """
    Display the memory usage of a dataframe by type.

    Example: mem_usage(data0, ['float','int','object'])
    """
    if isinstance(pandas_obj, pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else:  # we assume if not a df it's a series
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2  # convert bytes to megabytes
    print("\n{:03.2f} MB".format(usage_mb))
        # same as sys.getsizeof (pandas_obj)/1024**2

    for dtype in list_typ:
        selected_dtype = pandas_obj.select_dtypes(include=[dtype])
        sum_usage_b = selected_dtype.memory_usage(deep=True).sum()
        sum_usage_mb = sum_usage_b / 1024 ** 2
        print("Total memory usage for {} columns: {:03.2f} MB"
              .format(dtype, sum_usage_mb))



def mydisplay(df, title='', gen_width=100, gens=None, dig=2, digs=None):
    """
    Pretty display of a DataFrame.

    gen_width is the width in pixels for all columns
    gens is a dictionary giving specified width for some columns
       (ex: gens['colname']=60)
    dig is the number of digits after the dot for all numeric columns
    digs is a dictionary giving specified dig for some columns
    """
    print('\n', title, ' - ', len(df), 'lines:')
    if isinstance(df, pd.Series):
        df = df.to_frame()
    dfs = (df.style
             .set_table_styles([dict(selector="table, th, td",
                                     props=[('border', '1px solid indigo'),
                                            # {N}px solid {color}
                                            ('border-collapse', 'collapse'),
                                            # or ('border-color', '#96D4D4')
                                            ])], overwrite=False)
           #              # To change the background one row out of two
           #              .set_table_styles([dict(selector="tr:nth-child(even)",
           #                                      props=[('background-color',
           #                                                  '#0a3948'),
           #                                             ])], overwrite=False)
           # Header
             .set_table_styles([dict(selector="th",
                                     props=[('text-align', 'center'),
                                            ('word-break', 'break-all'),
                                            ('min-width', str(gen_width) +
                                             'px'),  # '120px' or '30em'
                                            ('max-width', str(gen_width) +
                                             'px'),
                                            ('overflow', 'hidden'),
                                            ('text-overflow', 'ellipsis'),
                                            ('white-space', 'nowrap'),
                                            ])], overwrite=False)
           # Data (cells)
             .set_table_styles([dict(selector="td",
                                     props=[('text-align', 'center'),
                                            ('word-break', 'break-all'),
                                            # ('word-wrap', 'break-word')
                                            ('min-width', str(gen_width) +
                                             'px'),
                                            ('max-width', str(gen_width) +
                                             'px'),
                                            ('overflow', 'hidden'),
                                            # necessary to cut text when
                                            #   arrives at column width
                                            ('text-overflow', 'ellipsis'),
                                            ('white-space', 'nowrap'),
                                            ])], overwrite=False)
           #              #To change only selected columns (may need to do the
           #              #  same for th also)
           #              if gens is not None:
           #                    for col in gens.keys:
           #                      .set_table_styles({col :[dict(selector='td',
           #                                              props=[('min-width',
           #                                              str(gens[col]) + 'px'),
           #                                                     ('max-width',
           #                                                      str(gens[col])
           #                                                          + 'px'),
           #                                                   ])]}, overwrite=False)
           # To show the full text when the mouse hover a cut text
             .set_table_styles([dict(selector="td:hover",
                                     props=[('overflow', 'visible'),
                                            ('white-space', 'unset'),
                                           ])], overwrite=False)
             .set_table_styles([dict(selector="th:hover",
                                     props=[('overflow', 'visible'),
                                            ('white-space', 'unset'),
                                           ])], overwrite=False)
           #              .set_properties(subset=df.columns[0:3].values,
           #                              **{'width': '10px'},
           #                              **{'max-width': '10px'})
           # If not overwrite=False, the parameters set before are deleted
           # For color code like '#2E2E2E' :
           #   https://htmlcolorcodes.com/fr/selecteur-de-couleur/
           # For color name like 'indigo':
           #   https://www.computerhope.com/jargon/w/w3c-color-names.htm

           # .set_properties(**{'white-space': 'wrap'})
           # .set_properties(subset=['url'], **{'width': '46px'})#,
           #                 **{'max-width:'46px'})   # for only some columns
           # num_col_mask = df.dtypes.apply(lambda d: issubclass(np.dtype(d).type,
           #                      np.number)) / subset=df.columns[num_col_mask]
           # #To change only selected columns (may need to do the same
           # #  for th also)
    )

    if gens is not None:
        for col in gens.keys():
            dfs = dfs.set_table_styles({col : [dict(selector='td',
                                                    props=[('min-width',
                                                            str(gens[col]) +
                                                            'px'),
                                                           ('max-width',
                                                            str(gens[col]) +
                                                            'px'),
                                                        ])]}, overwrite=False)

    dfs = dfs.format(precision=dig)
    display(dfs)
    print('\n')




# def sep_func():
#     '''Display a separator line & the function name at the start
#           of a function
#     Removed here as it gives an error on 'sys name' when imported
#     '''

#     print(nl, 50*'-', sys._getframe().f_back.f_code.co_name, nl)
#     #print(nl, 50*'-', sys._getframe().f_code.co_name, nl)
#     a = 1




def ordered_boxplot(x, y, data):
    """
    Draw a boxplot of data[x] by data[y] ordered according to the
      mean value of the boxes.

    x and y are column names from dataframe data
    The count of each modality is added to y tick labels, as well
       as a marker for the mean of each box
    """
    grouped = (data.loc[:, [x, y]]
                   .groupby(y)
                   .agg({x: ['median', 'count']})
                   .sort_values(by=(x, 'median'), ascending=False)
              )
    fig, ax = plt.subplots(figsize=(15, 11))
    g = sns.boxplot(x=x, y=y, data=data, ax=ax, order=grouped.index,
                    showmeans=True,  # , showfliers=False, orient='h'
                    meanprops={"marker": "s", "markerfacecolor": "white",
                               "markeredgecolor": "blue", "markersize": "3"})
    # g.set_xticklabels(g.get_xticklabels(), rotation=45, fontsize=10,
    #  horizontalalignment='right')
    ax.set_yticklabels([ind + ' (' + str(grouped.loc[ind, (x, 'count')]) +
                        ')' for ind in grouped.index])
                        # , fontdict={'fontsize': 7})
    plt.show()
#     print([ind + ' (' + str(grouped.loc[ind, (x, 'count')]) + ')'
#            for ind in grouped.index])




def mysound():
    """Play a sound (when program finishes, when error, ...)."""
    play_time_seconds = 1  ; framerate = 4410
    t = np.linspace(0, play_time_seconds, framerate * play_time_seconds)
    audio_data = np.sin(2 * np.pi * 300 * t) + np.sin(2 * np.pi * 240 * t)
    display(Audio(audio_data, rate=framerate, autoplay=True))



def custom_exc(shell, etype, evalue, tb, tb_offset=None):
    """Use as the function is called on exceptions in any cell."""
    shell.showtraceback((etype, evalue, tb), tb_offset=tb_offset)
    mysound()

# # Play a sound when any error occurs
# get_ipython().set_custom_exc((Exception,), custom_exc)  # used in IPython




def mygarbage():
    """Manual garbage collection (but doesn't seem to be effective)."""
    print(nl * 3, "Garbage Collection", nl, "threshold:", gc.get_threshold())
    # gc.set_threshold()  #to get/set the threshold of automatic
    #                     #  garbage collection for the 3 generations
    print("len(get_objects):", len(gc.get_objects()))
    print("stats:", gc.get_stats())
    print("count:", gc.get_count())  # number of objects for each generation
    # gc.garbage
    print("nb of collected:", gc.collect(), '\n')  # manual garbage collection
    print("len(get_objects):", len(gc.get_objects()))
    print("stats:", gc.get_stats())
    print("count:", gc.get_count(), nl * 3)  # number of objects for each gen.

mygarbage()




# Quick function to pretty print & format numbers in a list
def f_pr0(list1):
    """."""
    return print(' - '.join('{:10,.0f}'.format(f)
                            .replace(',', ' ') for f in list1))

def f_pr3(list1):
    """."""
    return print(' - '.join('{:10,.3f}'.format(f)
                            .replace(',', ' ') for f in list1))



def namestr(obj, namespace):
    """
    Get the name (as a string) of the variable.

    Example: namestr(a, globals())[0]
    """
    return [name for name in namespace if namespace[name] is obj]



def printb(*args):
    """Print a list of variable names and their values."""
    for s in args:
        #         print(namestr(s, locals())[0] + ': ')
        print(namestr(s, globals())[0] + ': ' + str(s))
        print('Length:', len(s), nl)



# Plot all numerical and categorical variables in a df
def project_plot(data0, x_log0=True, y_log0=True):
    """
    Plot to explore numerical and categoricaldata and look for outliers.

    x_log & y_log to choose the scale of the plots on numeric
       variables (easier to find outliers)
    """
    data0 = data0.copy()
    max_mod = 30  # max number of modalities displayed per categorical var.

    # General data histograms
    data0.hist(figsize=(15, 14), bins=30)
    plt.tight_layout()
    plt.show()

    # # Pairplot (takes several minutes)
    # sns.pairplot(data0, hue='BuildingType', corner=True)
    # plt.savefig('./pairplot5.png', dpi=400)
#     lcol = list(data0.select_dtypes(include=[np.number]).columns)
#     sns.pairplot(data0, vars=lcol, kind='reg', diag_kind='kde',
#                  plot_kws={'scatter_kws': {'alpha': 0.1}})
#     plt.show()


    # Plot numeric (and date) data to see if outliers (use log on x and y)
    num_cols = data0.select_dtypes(include=['number', 'datetime64[ns]']
                                  ).columns
    for c in num_cols:
        print(nl, c)
        x_log, y_log = x_log0, y_log0
        if data0[c].dtype == 'datetime64[ns]':
            x_log = False
            format1 = "%Y/%m/%d"
            data_date = pd.to_datetime(data0[c]).dt.date  # dtype=object
                # if dt.date   or better use dt.floor('d') to keep type
            val1 = pd.to_datetime(np.unique(data_date
                                            .sort_values()
                                            .values)[:15])
            val2 = pd.to_datetime(np.unique(data_date
                                            .sort_values()
                                            .values)[-15:][::-1])
            print(' unique min =', [x.strftime(format1) if x == x else 'NaN'
                                    for x in val1])
            print(' unique max =', [x.strftime(format1) if x == x else 'NaN'
                                    for x in val2])
        else:
            print(' unique min =', [round(x, 1) if (abs(x) >= 1)
                                    else round(x, 3)
                                    for x in np.unique(data0[c]
                                                       .sort_values()
                                                       .values)[:15]])
            print(' unique max =', [round(x, 1) if (abs(x) >= 1)
                                    else round(x, 3)
                                    for x in np.unique(data0[c]
                                                       .sort_values()
                                                       .values)[-15:][::-1]])
        # data0b = data0.loc[data0[c]>0, c]  #remove value 0 just for
        # plotting (for a better zoom)
        # to update if values negative possible
        data0b = data0[c]
        fig, ax = plt.subplots(figsize=(15, 7))
        x_min, x_max = data0b.min(), data0b.max()
#         if data0b.dtype != 'datetime64[ns]':
#             print('***', x_log, ((x_max-x_min)>1000))
        if x_min == 0:
            x_min = 1e-5
        if x_log and ((x_max - x_min) > 100):
            bins = np.geomspace(x_min, x_max, 250)  # error if 0 in geomspace
            ax.set_xscale('log')
        else:
            bins = 250

        data0b.hist(bins=bins, log=y_log)

        ax.tick_params(axis='both', which='minor', bottom=True, left=True)
        ax.tick_params(axis='both', which='major', bottom=True, left=True)

        myxticks = [x for x in ax.get_xticks(minor=True)
                    if (x >= x_min) * (x <= x_max)]
        myxlabels = [str(x) for x in myxticks]
        ax.set_xticks(myxticks, minor=True)  # to avoid warning (xticks are
            # the location and xticklabels their labels)
        ax.set_xticklabels(myxlabels, rotation=30, fontsize=8, minor=True,
                           horizontalalignment='right')

        ax.set_xticks(ax.get_xticks())  # to avoid warning
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, fontsize=8,
                           horizontalalignment='right')

        if data0[c].dtype != 'datetime64[ns]':
            print(data0[c].dtype)
#             format1 = format(x)
#         elif x_max >= 100:
#             format1 = format(round(x, 1), ',')
#         else:
#             format1 = format(round(x, 2), ',')
            nb = 1 if x_max >= 100 else 2
            ax.get_yaxis().set_major_formatter(mpl.ticker.FuncFormatter(
                lambda x, p: format(round(x, nb), ',')))  # int(x)
            ax.get_xaxis().set_major_formatter(mpl.ticker.FuncFormatter(
                lambda x, p: format(round(x, nb), ',')))
            ax.get_yaxis().set_minor_formatter(mpl.ticker.FuncFormatter(
                lambda x, p: format(round(x, nb), ',')))
            ax.get_xaxis().set_minor_formatter(mpl.ticker.FuncFormatter(
                lambda x, p: format(round(x, nb), ',')))
            ax.set_xlim(x_min, x_max)  # must be at the end to get only
                # plot for [x_min, x_max]
        plt.show()



    # Plot categorical data (count plot)
    for c in data0.select_dtypes(include=['object']).columns:
        # trick because miss the case where we have like [1] which
        #  is not considered as a list in pyhton
        if 1:  # (str(data0.iloc[1].loc[c])[0] != '[') and
            # (sum(data0[c].isna())==0):
            print(nl, c)
            # if it contains a list, simply take the 1st element
            #   (often there is only one element in the list)
            data0[c] = data0[c].map(lambda x: x[0] if isinstance(x, list)
                                    else x)
            col_order = data0[c].value_counts().index
            col_order = col_order[0:min(max_mod, len(col_order))]
            mydisplay(data0[c].head())
            print(type(col_order), col_order[:10])
            fig, ax = plt.subplots(figsize=(15, 0.35 * len(col_order)))
            sns.countplot(y=c, order=col_order, data=data0)

            # Print the value next to each bar
            xmin, xmax = plt.xlim()
            for p in ax.patches:
                width = p.get_width()  # get bar length
                ax.text(width + np.sqrt(xmax - 3) / 10,
                            # set the text just after the bar
                        p.get_y() + p.get_height() / 2,
                            # get Y coordinate + X coordinate / 2
                        '{:1.0f}'.format(width), ha='left', va='center'
                            # vertical alignment
                )
            plt.show()

#         # Display the counts for each modality
#         with pd.option_context('display.max_rows', None):
#             mydisplay(data0[c].value_counts().sort_index()
#                               .reset_index().head(100))



def mytimer(message):
    """
    Multi-Chronometer from start_time.

    Use first start_time = time.time()
    """
    global start_time
    # sep_func()
    print(f"\n**** {time.time() - start_time:.2f} seconds --- {message}\n")


@contextmanager
def timer(title):
    """
    Give the duration in secondsof a group of operations.

    Exemple: with timer("Process credit card balance"):
        cc = credit_card_balance(idx=df['SK_ID_CURR'])
        ...
        del cc ; gc.collect()
    """
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))
