import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
print(os.getcwd())
starttime = time.time()


# Helper methods
def filter_pfand(df):
    return df.query('wg_bez != "Pfand"')


def ratio(df, enumerator, nominator):
    return df[enumerator]/df[nominator]


def ratio_rows(df, enumerator, nominator):
    return df.loc[enumerator]/df.loc[nominator]


def load_and_merge_files(path=os.getcwd()):
    '''
        Load all csv files in given directory and merges them
    '''
    # Initializing
    df_list = []
    filelist = os.listdir(path)
    csv_files = []
    dataframe = pd.DataFrame()

    # Subsets filelist to csv only
    for file in filelist:
        if file.endswith('.csv'):
            csv_files.append(file)

    # read each csv file and append it to the DataFrame
    for csv_file in csv_files:

        df_list.append(pd.read_csv('{path}/{filename}'.format(path=path,
                                                              filename=csv_file),
                                   sep=';'
                                   )
                       )
    dataframe = pd.concat(df_list, ignore_index=True, verify_integrity=True)

    return dataframe


def typecasting(df, object_cols=[], numeric_cols=[], date_cols=[]):
    '''
        retype columns of a dataframe

    '''
    df[object_cols] = df[object_cols].astype('object')
    df[date_cols] = df[date_cols].astype('datetime64[s]')
    df[numeric_cols].apply(pd.to_numeric)

    return df


def data_cleaning(df):
    '''
        Adding coupon_betrag to umsatz_brutto for real values
        Dropping coupon values as they are not targeted in this analysis
        Dropping arthiertyp_id as its constant
        Recalculate bon_ends to duration as it provides no new information
        Filter 'Pfand' entries
    '''
    return (df.rename(str.lower, axis='columns').
            assign(umsatz_brutto=df['umsatz_brutto'].
                   add(df['coupon_betrag'], fill_value=0)).
            drop(['coupon_menge',
                  'coupon_kz',
                  'coupon_zeile',
                  'coupon_id',
                  'coupon_betrag',
                  'arthiertyp_id'],
                 axis='columns').
            pipe(typecasting, date_cols=['bon_beginn', 'bon_ende'],
                 object_cols=['kl_art_id', 'wgi_id', 'wg_id']).
            assign(bon_ende=lambda x: x['bon_ende']-x['bon_beginn']).
            rename(columns={'bon_ende': 'bon_duration'}).
            pipe(filter_pfand)
            )

# Load Data
df = (load_and_merge_files('./Kassendaten').
      pipe(data_cleaning)
      )

# initialize dicts for results
results_wg = {}
results_overall = {}

# Was sind die 20 größten Warengruppen?

# get top 20 'warengruppe'
wg_umsatz = (df.groupby(['wg_bez']).
             sum().
             loc[:, ['umsatz_brutto']]
             )

#  it into results dict
results_wg.update({'wg_umsatz': wg_umsatz})

# get_top20
top20_wg = wg_umsatz.sort_values('umsatz_brutto', ascending=False).head(20)

# calculate avg_umsatz as reference
avg_wg_umsatz = wg_umsatz.mean()

# store into overall resutls
results_overall.update({'avg_wg_umsatz': avg_wg_umsatz[0]})

# plot barchart
fig, ax = plt.subplots(figsize=(7, 5))
title = "Top 20 Umsatz nach Warengruppe"
top20_wg.plot.bar(ax=ax, title=title)
ax.axhline(avg_wg_umsatz.values, label="Durchschnitt", color='black')
ax.set(xlabel='Warengruppe')
ax.legend()
fig.savefig(f'{title}.png', dpi=300)

# Wie viel wird im Schnitt in einer Warengruppe ausgegeben ###
avg_wg_value_on_bon = (np.divide(df['umsatz_brutto'].sum(),
                                 (df.groupby('bon_id')['wg_bez'].
                                  nunique().
                                  sum()
                                  )
                                 )
                       )

# Store into results
results_overall.update({'avg_wg_value_on_bon': avg_wg_value_on_bon})

# Wie viel wird im Schnitt innerhalb einer Warengruppe gekauft wenn sie gekauft wird
avg_wg_value_each_wg = (df.groupby('wg_bez').
                        aggregate({'bon_id': 'nunique', 'umsatz_brutto': 'sum'}).
                        rename(columns={'umsatz_brutto': 'sum_umsatz_per_wg',
                                        'bon_id': 'count_bon_id'}).
                        pipe(ratio, enumerator='sum_umsatz_per_wg', nominator='count_bon_id')
                        )
# store into results
results_wg.update({'avg_wg_value_on_bon': avg_wg_value_each_wg})

# wie viele Artikel werden im schitt zusammen gekauft
avg_nbr_of_article_on_bon = df.groupby('bon_id')['kl_art_id'].nunique().mean()

# store into results
results_overall.update({'avg_articles_on_bon': avg_nbr_of_article_on_bon})

# Was ist der mittlerer bonwert im verlauf eines Monats/Woche
avg_bon_value_month = (df.groupby([df.bon_beginn.dt.day, 'bon_id'])
                       ['umsatz_brutto'].
                       sum().
                       mean(level='bon_beginn')
                       )

avg_bon_value_week = (df.groupby([df.bon_beginn.dt.dayofweek, 'bon_id'])
                      ['umsatz_brutto'].
                      sum().
                      mean(level='bon_beginn')
                      )

# store into results overall
# peak to peak
results_overall.update({'ptp_bon_values_over_month': avg_bon_value_month.pipe(np.ptp),
                       'ptp_bon_values_over_week': avg_bon_value_week.pipe(np.ptp)})

# plotting results
# Weekly
fig, ax = plt.subplots(figsize=(7, 5))
title = "Fluktuation des AVG-Bonwerts (Woche)"
avg_bon_value_week.plot.bar(ax=ax,
                            title="Fluktuation des AVG-Bonwerts (Woche)",
                            label='AVG Bonwert')
# reference line for minimum and maximum
ax.axhline(avg_bon_value_week.min(), label="Min", color='black', linestyle='--')
ax.axhline(avg_bon_value_week.max(), label="Max", color='black', linestyle='--')
ax.set(xlabel='Wochentag')
ax.legend()
fig.savefig(f'{title}.png', dpi=300)

# monthly
fig, ax = plt.subplots(figsize=(7, 5))
title = "Fluktuation des AVG-Bonwerts (Monat)"
avg_bon_value_month.plot.bar(ax=ax,
                             title=title,
                             label='AVG Bonwert')
# reference line for minimum and maximum
ax.axhline(avg_bon_value_month.min(), label="Min", color='black', linestyle='--')
ax.axhline(avg_bon_value_month.max(), label="Max", color='black', linestyle='--')
ax.set(xlabel='Tag')
ax.legend()
fig.savefig(f'{title}.png', dpi=300)

# werden bestimmte warengruppen besonders oft zu beginn eines Monats/Woche gekauft


def get_freq_table(df, target, relative=False):
    '''
        calculates a frequency table for each wg per targeted timespan
        in comparrison of the total amount of bons on that day so e.g.
        weekends where in general are absolutely more customers do not
        bias the estimation
    '''
    total_bons_per_target = (df.groupby([target])
                             ['bon_id'].
                             nunique())

    return (df.groupby(['wg_bez', target])['bon_id'].
            nunique().
            divide(total_bons_per_target, level='bon_beginn').
            unstack('wg_bez')
            )


def ratio_beginn_end(df, bins):
    '''
        returns the ratio between begin and end of the timespan
        bins defines where the split between beginn and end is
    '''
    return (df.reset_index().
            assign(bon_beginn=lambda x: pd.cut(x['bon_beginn'],
                                               bins=bins,
                                               labels=['beginn', 'end'],
                                               include_lowest=True)).
            groupby('bon_beginn').
            mean().
            pipe(ratio_rows, enumerator='beginn', nominator='end').
            sort_values(ascending=False)
            )


# calculate relative frequency table for each wg per day
freq_table_count_wg_week = df.pipe(get_freq_table,
                                   target = df.bon_beginn.dt.dayofweek)
freq_table_count_wg_month = df.pipe(get_freq_table,
                                    target = df.bon_beginn.dt.day)

# calculate the ratio between beginn and end for week and month
ratio_beginn_end_week = freq_table_count_wg_week.pipe(ratio_beginn_end,
                                                      bins = [0,1,5])
ratio_beginn_end_month = freq_table_count_wg_month.pipe(ratio_beginn_end,
                                                        bins = [0,4,30])

#support for ratio between beginn and end
#too less observations would bias estimation
days_with_wg = (df.assign(bon_beginn = lambda x: x['bon_beginn'].dt.date).
                groupby(df['wg_bez'])['bon_beginn'].
                nunique().
                div(df['bon_beginn'].dt.date.nunique())
)



#store into results
results_wg.update({'ratio_beginn_end_week':ratio_beginn_end_week,
                  'ratio_beginn_end_month': ratio_beginn_end_month,
                  'bon_per_wg_overall': days_with_wg})

# plot top wg with highes ration between begin and end 
# Wochenverlauf
fig, ax = plt.subplots(figsize =(7,5))
title = 'Relative Häufigkeit gekaufter Warengruppen im Wochenverlauf'
top5 = ratio_beginn_end_week.nlargest(7).index.values
freq_table_count_wg_week[top5].plot(ax=ax,title=title)
ax.set(xlabel='Tag',ylabel='Anteil Bons mit Warengruppe')
# legend outside the plot
ax.legend(title='Warengruppe',loc='center left', bbox_to_anchor=(1, 0.5))
fig.tight_layout()
fig.savefig(f'{title}.png',dpi=300)

# monatsverlauf
fig, ax = plt.subplots(figsize =(7,5))
title = 'Relative Häufigkeit gekaufter Warengruppen im Monatsverlauf'
top5 = ratio_beginn_end_month.nlargest(5).index.values
ratio_beginn_end_month.nlargest(5).index.values
freq_table_count_wg_month[top5].plot(ax=ax,title=title)
ax.set(xlabel='Tag',ylabel='Anteil Bons mit Warengruppe')
fig.savefig(f'{title}.png',dpi=300)

# Welche lebensmittel haben die größte Umsatzeffizienz
# Revenue efficency per Food article 
revenue_efficency_per_food = (df.loc[df['wg_id']<=430].  #foodstuff is id <= 430
                              groupby('kl_art_id').
                              agg({'umsatz_brutto':'sum','bon_id':'nunique'}).
                              pipe(ratio,enumerator='umsatz_brutto',nominator='bon_id').
                              round(2)                              
)

# support for revenue efficency, wie häufig erschien ein artikel auf einem Bon
abs_frequ_art = (df.loc[df['wg_id']<=430]. #foodstuff is id <= 430
                 groupby('kl_art_id').
                 agg({'bon_id':'nunique',
                      'wgi_bez': lambda x: stats.mode(x)[0]})

)

revenue_efficency= (pd.concat([revenue_efficency_per_food, abs_frequ_art],
                              sort=True,
                              axis=1).
                    rename({0:'efficency','bon_id':'frequency'},
                           axis=1)
                   )

# Plot Umsatzeffizient

# get top 5 efficient
top5_efficency = revenue_efficency.nlargest(5,columns='efficency')
top5_efficency.to_clipboard()
fig, [ax1,ax2] =plt.subplots(1,2,figsize=(15,5))
title= 'Umsatzeffizienz'
fig.suptitle(title, fontsize=14)
# plot char in the left
revenue_efficency_per_food.plot(ax=ax1)
ax1.set_ylabel('Umsatzeffizienz')

# Print support table on right axis
ax2.axis('tight')
ax2.axis('off')
table = ax2.table(cellText=top5_efficency.values,
                  colLabels=top5_efficency.columns,
                  loc='center')
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.2,2)
# plot graph
fig.savefig(f'{title}.png',dpi=300)

# plot Effizienz vs Häufigkeit
fig, ax = plt.subplots(figsize =(7,5))
title='Effizient vs. Häufigkeit'
revenue_efficency.plot.scatter(x='efficency',y='frequency',
                               ax=ax,
                               title=title)
ax.set_yscale('log')
ax.set_xlabel('Umsatzeffizienz')
ax.set_ylabel('Log-Häufigkeit eines Artikels')
fig.savefig(f'{title}.png',dpi=300)

# Export to Excel
# summarize result dict
sheet1 = pd.concat(results_wg,axis=1, sort=True).round(2)
sheet2 = pd.Series(results_overall).to_frame()
writer = pd.ExcelWriter('results.xlsx', engine='xlsxwriter')

# Convert the dataframe to an XlsxWriter Excel object.
sheet1.to_excel(writer, sheet_name='WG Results')
sheet2.to_excel(writer, sheet_name='Question Answers')

# Close the Pandas Excel writer and output the Excel file.
writer.save()

# Bonus Zeitreihendaten

art_per_bon = df.groupby(['bon_beginn', 'bon_id'])['kl_art_id'].nunique()

# Plotting
fig, ax = plt.subplots(figsize =(10,5))
title = 'Durchschnittliche Warenkorbgröße im Zeitverlauf'
art_per_bon.resample('D',level='bon_beginn').mean().dropna().plot(ax=ax, alpha=0.8,label='Tagesverlauf')
art_per_bon.resample('H',level='bon_beginn').mean().rolling('3h').mean().plot(ax=ax,alpha=0.8,label='Stundenverlauf')
art_per_bon.resample('W',level='bon_beginn').mean().dropna().plot(ax=ax,alpha=0.8,label='Wochenverlauf')
fig.suptitle(title, fontsize=14)
ax.set_ylabel('Anzahl versch. Artikel pro Warenkorb')
ax.legend()

fig.savefig(f'{title}.png', dpi=300)

# durchschittliche artikel pro warenkorb im tages bzw wochenverlauf
fig, ax = plt.subplots(figsize=(10, 5))
title = 'Durchschnittliche Warenkorbgröße über den Tag'
labels=['Mo', 'Di' ,'Mi' ,'Do' ,'Fr' ,'Sa']

(df.query('bon_beginn.dt.hour > 8').
 groupby([df.bon_beginn.dt.dayofweek,df.bon_beginn.dt.hour,'bon_id']).
 aggregate({'kl_art_id': 'nunique'}).
 mean(level=[0,1]).
 unstack(level=0).plot(ax=ax)
 )
ax.set_xlabel('Stunde')
ax.legend(labels=labels)
fig.suptitle(title, fontsize=14)
fig.savefig(f'{title}.png', dpi=300)

print("### Completed ###")
print("Runntime: ", (time.time()-starttime))
print('#################')