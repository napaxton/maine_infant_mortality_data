%pylab inline
import types
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FormatStrFormatter

# # set some plotting properties
fontP = FontProperties()
fontP.set_size('small')

import pandas as pd
from locale import *
setlocale(LC_NUMERIC, '')
np.set_printoptions(linewidth=800, threshold='nan', precision=5)

raw = pd.read_csv('data/us_infant_mortality_2.tsv', sep='\t', thousands=',', index_col=0)

def OAT(raw, the_one, how_much=1, sort_cols=None, sort_asc=None):
    
    if sort_cols is not None:
        raw = raw.sort(columns=sort_cols, ascending=sort_asc)
    
    year_set = set()
    for column in raw.columns:
        year_set.add(column.split('_')[0])

    years = list(year_set)
    years.sort()
    years.insert(0, 'states')

    rates = []

    states = list(raw.index)

    for state in states:
        row = [state]
        for year in years[1:]:

            # get the birth value
            try:
                b = raw['{}_Births'.format(year)][state]
            except KeyError:
                continue

            # get death value
            if state == the_one:
                delta = how_much
            else:
                delta = 0
                
            try:
                d = atof(raw['{}_Deaths'.format(year)][state]) + delta
            except AttributeError:
                d = raw['{}_Deaths'.format(year)][state] + delta
                
            d = 0 if d < 0 else d
            
            row.append(1.0*d/b)

        rates.append(row)

    # The completed dataframe of mortality rates
    rates = pd.DataFrame.from_records(rates, columns=years, index='states')

    # Compute numerical data ranks (1 through n) among the states for each year
    ranks = rates.rank(axis=0)
    
    return (rates, ranks)

def cohens_d(means=(), stdevs=(), N_samples=()):
    """
    cohens_d giving the effect size between group 1 and group 2

    see http://en.wikipedia.org/wiki/Effect_size#Cohen.27s_d

    :param means: tuple with (M1,M2)
    :param stdevs: tuple with (V1,V2)
    :param N_samples: tuple with (N1,N2)
    :return: float, effect size
    """

    m = means
    s = stdevs
    n = N_samples
    S = sqrt(((n[0] - 1) * s[0] ** 2 + (n[1] - 1) * s[1] ** 2) / (n[0] + n[1] - 2))

    return (m[0] - m[1]) / S
    
rates, ranks = OAT(raw, 'Maine', how_much=0)
rates_plus, ranks_plus = OAT(raw, 'Maine', how_much=10)
rates_minus, ranks_minus = OAT(raw, 'Maine', how_much=-10)

base = ranks.loc['Maine'].to_frame(name='base')
plus = ranks_plus.loc['Maine'].to_frame(name='plus')
minus = ranks_minus.loc['Maine'].to_frame(name='minus')

maine = base.merge(plus, left_index=True, right_index=True)
maine = maine.merge(minus, left_index=True, right_index=True)

maine

def area_between_curves(df):
    """df is a dataframe with 'base', 'plus', and 'minus' columns
    and year index values
    """
    running = []
    years = list(df.index)
    for year in years:
        plus = df['plus'][year]
        minus = df['minus'][year]
        running.append(plus - minus)
        
    return sum(running)

def average_change(df):
    """df is a dataframe with 'base', 'plus', and 'minus' columns
    and year index values
    """
    plus = []
    minus = []
    years = list(df.index)
    total = len(years)
    
    for year in years:
        bs = df['base'][year]
        up = df['plus'][year]
        dn = df['minus'][year]
        plus.append(up - bs)
        minus.append(bs - dn)
        
    hi = 1.0*sum(plus)/total
    low = 1.0*sum(minus)/total
        
    return (hi, low)
    
states = list(raw.index)

S = []

for state in states:
    rates_plus, ranks_plus = OAT(raw, state, how_much=10)
    rates_minus, ranks_minus = OAT(raw, state, how_much=-10)

    base = ranks.loc[state].to_frame(name='base')
    plus = ranks_plus.loc[state].to_frame(name='plus')
    minus = ranks_minus.loc[state].to_frame(name='minus')

    df = base.merge(plus, left_index=True, right_index=True)
    df = df.merge(minus, left_index=True, right_index=True)

    A = area_between_curves(df)
    
    hi, low = average_change(df)
    
    S.append([state, A, hi, low])
    
sensitivity = pd.DataFrame.from_records(S, columns=['states','sensitivity', 'high', 'low'], index='states')
sensitivity = sensitivity.sort(columns=['high', 'low', 'sensitivity'])
sensitivity

mask = rates.index.isin(['Maine'])
rates_all_others = rates[~mask]

mask = rates.index.isin(['Maine'])
rates_maine_only = rates[mask]

def bootstrap(data, n_samples, iterations=100):
    
    means = []
    
    mn = data.min()
    mx = data.max()
    
    for i in xrange(iterations):
        samples = np.random.choice(data, size=n_samples, replace=True)
        mean = means.append(np.mean(samples))
        
    means = np.array(means)
    
    hi = np.percentile(means, 97.5)
    lo = np.percentile(means, 2.5)
    mu = np.mean(means)
    
    return (lo, mu, hi, mn, mx)

def m_histogram(data, my_bins=10, title=None, xlim=None, ylim=None):

    if not isinstance(data, types.StringType):

        fig = plt.figure(figsize=(8, 6))
        ax = plt.subplot(111)

        if isinstance(data, types.DictionaryType):
            for k, v in data.iteritems():
                n, bins, patches = ax.hist(v,
                                           bins=my_bins, normed=0,
                                           alpha=0.75, label=k.title(), rwidth=1)

            # only show the legend if there are multiple arrays
            plt.legend(loc='upper right', prop=fontP, scatterpoints=1)

        else:
            n, bins, patches = ax.hist(data,
                                       bins=my_bins, normed=0,
                                       alpha=0.75, rwidth=1)

    else:
        raise TypeError('Input is either an iterable of numbers, or dict of iterables')

    bin_centers = 0.5 * np.diff(bins) + bins[:-1]

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_ylabel('Frequency')
    if title is not None:
        plt.title(title)
    plt.grid(True)
    plt.subplots_adjust(bottom=0.15)
    return n, bins, fig

my_bins = np.arange(1, 20, 1)
ylim = (0, 25)
us = (1000*(rates_all_others.values)).flatten()
me = (1000*(rates_maine_only.values)).flatten()
from collections import OrderedDict
source = OrderedDict((('us', us), ('me', me)))
n, bins, h = m_histogram(source, my_bins=my_bins, title='Incoming')
h.savefig('tmp.png')

rates, ranks = OAT(raw, 'none', how_much=0, sort_cols=['2014_Births'], sort_asc=[False])

first_half = [str(i) for i in range(1995, 2005)]
second_half = [str(i) for i in range(2005, 2015)]
rates_first = rates[first_half]
rates_second = rates[second_half]

fig = plt.figure(num=None, figsize=(6, 12), dpi=1200, facecolor='w', edgecolor=None)

frame = plt.gca()
ax = fig.add_subplot(111)
# frame.axes.get_xaxis().set_visible(False)
frame.axes.get_yaxis().set_visible(False)

xmin = 999
xmax = 0

for rdx, R in enumerate([rates_first, rates_second]):
    states = R.index
    for idx, state in enumerate(states):
        mask = R.index.isin([state])
        state_rates = R[mask]

        n_samples = 20
        lo, mu, hi, mn, mx = bootstrap((1000*state_rates.values).flatten(), n_samples, iterations=100)

        width = mx - mn

        xmin = mn if mn < xmin else xmin
        xmax = mx if mx > xmax else xmax

        if state == 'Maine':
            sz = 15
            if rdx > 0:
                clr = 'red'
            else:
                clr = "none"
        else:
            sz = 10
            if rdx > 0:
                clr = 'blue'
            else:
                clr = "none"
                
        if rdx > 0:
            offset = 0
        else:
            offset = 0.2

        ax.add_patch(
            matplotlib.patches.Rectangle(
                (mn, idx-offset),   # (x,y)
                width,          # width
                0.5,          # height
                facecolor=clr, 
                
            )
        )

        ax.text(17, idx+0.25, state, ha="left", va="center", size=sz)

ax.annotate('', xy=(17, 0), xycoords='axes fraction', xytext=(1, -0.1), 
            arrowprops=dict(arrowstyle="<->", color='k'))

plt.annotate('Increasing number of births',(22, 10),(22, 30),va='center',ha='center', 
               rotation=270, arrowprops=dict(arrowstyle='->'), annotation_clip=False) 

this_decade = matplotlib.patches.Rectangle((0, 2), 1, 1, facecolor='blue')
last_decade = matplotlib.patches.Rectangle((0, 1), 1, 0.25, facecolor='none')
plt.legend([this_decade, last_decade], ['2005-2014', '1995-2004'], loc='lower right')
plt.ylim(-1,53)
plt.xlim(xmin-0.5, xmax+0.5)
plt.title('Ranges of Infant Deaths for each State\n1995-2004 vs. 2005-2014')
plt.xlabel('Number of Infant Deaths per 1000 Births')
# plt.tight_layout()
plt.savefig('state_ranges.jpg')
plt.show()

rates, ranks = OAT(raw, 'none', how_much=0)

fig = plt.figure(num=None, figsize=(750/96., 450/96.), dpi=96, facecolor='w', edgecolor=None)
ax = fig.add_subplot(111)

x = range(1995, 2015)

for state in ranks.index:
    
    mask = ranks.index.isin([state])
    st_ranks = ranks[mask]
    y = map(int, list(st_ranks.values.flatten()))
    
    new_england = ['Maine', 'New Hampshire', 'Massachussets', 'Vermont']
    
    if state in ['Maine']:
        ax.plot(x, y, label=state, color='r', linewidth=3)
    else:
        ax.plot(x,y, color='k', alpha=0.5)
plt.ylabel('State Rank')
plt.xlim(1995,2014)
plt.ylim(0,52)
plt.title("Maine's Infant Mortality Rank")
plt.gca().invert_yaxis()

labels = x

plt.xticks(x, labels, rotation=45)

plt.savefig('maines_rank.jpg')
plt.show()

rates, ranks = OAT(raw, 'none', how_much=0)

fig = plt.figure(num=None, figsize=(750/96., 450/96.), dpi=96, facecolor='w', edgecolor=None)
ax = fig.add_subplot(111)

x = range(1995, 2015)

for state in rates.index:
    
    mask = rates.index.isin([state])
    st_rates = rates[mask]
    y = list((1000*st_rates.values).flatten())
    
    new_england = ['Maine', 'New Hampshire', 'Massachussets', 'Vermont']
    
    if state in ['Maine']:
        ax.plot(x, y, label=state, color='r', linewidth=3)
    else:
        ax.plot(x,y, color='k', alpha=0.5)
plt.ylabel('State Rate (per 1000 births)')
plt.xlim(1995,2014)
plt.title("Maine's Infant Mortality Rate")
labels = x

plt.xticks(x, labels, rotation=45)

plt.savefig('maines_rate.jpg')
plt.show()

maine_rates = rates[rates.index.isin(['Maine'])]
mr = maine_rates.values.flatten()
np.std(mr)*1000

raw[['2010_Deaths', '2011_Deaths', '2010_Births', '2011_Births']]

# how much id Maine's infant mortality rate worsen in 2011?
print 1000*(85./12704 - 70./12970)

fig = plt.figure(num=None, figsize=(750/96., 450/96.), dpi=96, facecolor='w', edgecolor=None)
ax = fig.add_subplot(111)

x = range(1995, 2015)

births = df

import copy

births = copy.deepcopy(raw)
cols = births.columns
for col in cols:
    if 'Births' not in col:
        births.pop(col)

births
for state in births.index:
    
    mask = births.index.isin([state])
    st_rates = births[mask]
    y = map(int, list(st_rates.values.flatten()))
    
    small_states = [u'West Virginia', u'Hawaii', u'Maine', u'Montana', 
                    u'New Hampshire', u'South Dakota', u'Alaska', u'North Dakota', 
                    u'Delaware', u'Rhode Island', u'District of Columbia', u'Wyoming', 
                    u'Vermont']
    if state not in small_states:
        continue
    
    if state in ['Maine']:
        ax.plot(x, y, label=state, color='r', linewidth=3)
    else:
        ax.plot(x,y, color='k', alpha=0.5)
plt.ylabel('Number of Births')
plt.xlim(1995,2014)
plt.title("Total Births for the 13 Smallest States")
labels = x

plt.xticks(x, labels, rotation=45)

plt.savefig('maines_births.jpg')
plt.show()

fig = plt.figure(num=None, figsize=(750/96., 450/96.), dpi=96, facecolor='w', edgecolor=None)
ax = fig.add_subplot(111)

x = range(1995, 2015)

births = df

import copy

births = copy.deepcopy(raw)
cols = births.columns
for col in cols:
    if 'Births' not in col:
        births.pop(col)

deaths = copy.deepcopy(raw)
cols = deaths.columns
for col in cols:
    if 'Deaths' not in col:
        deaths.pop(col)
        
maine_births = births[births.index.isin(['Maine'])]
maine_deaths = deaths[deaths.index.isin(['Maine'])]

first_half = ['{}_Deaths'.format(str(i)) for i in range(1995, 2005)]
second_half = ['{}_Deaths'.format(str(i)) for i in range(2005, 2015)]
_first = maine_deaths[first_half]
_second = maine_deaths[second_half]

print _first.values.flatten().sum(), _second.values.flatten().sum()

y = map(int, list(maine_births.values.flatten()))
lns1 = ax.plot(x, y, label='Births', color='g', linewidth=3)

ax2 = ax.twinx()
y = map(int, list(maine_deaths.values.flatten()))
lns2 = ax2.plot(x, y, label='Deaths', color='r', linewidth=3)

lns = lns1+lns2
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=0, frameon=False)

plt.xlim(1995,2014)
plt.title("Births and Infant Deaths in Maine")
ax.set_ylabel("Number of Births")
ax2.set_ylabel('Number of Deaths')

labels = x

plt.xticks(x, labels, rotation='vertical')
plt.setp( ax.xaxis.get_majorticklabels(), rotation=45 )
plt.savefig('maine_births_deaths.jpg')
plt.show()

from scipy.stats import gamma, beta
from math import sqrt

def beta_params(m,s):
    '''convert mean and std to shape params for the beta distribution'''
    a = ((1-m)/(s**2) - 1/m)*m**2
    b = a*(1/m - 1)
    return a,b

def gamma_params(m,s):
    '''convert mean and std to shape params for the gamma distribution'''
    a = (m/s)**2
    b = (s**2)/m
    return a,b

def to_unity(values):
    '''convert a list of numbers to a list with the same proportions, but sums to unity'''
    s = 1.*sum(values)
    return [i/s for i in values]
    
# derive a theoretical gamma distribution from the U.S. IM rate data (this decade only)
rates, ranks = OAT(raw, 'none', how_much=0)

first_half = [str(i) for i in range(1995, 2005)]
second_half = [str(i) for i in range(2005, 2015)]
rates_first = rates[first_half]
rates_second = rates[second_half]

data = 1000*rates.values.flatten()

fit_alpha, fit_loc, fit_beta=gamma.fit(data)

# create the theoretical distribution by fitting to the data
us_gamma = gamma.rvs(fit_alpha, loc=fit_loc, scale=fit_beta, size=10000)

# compare the theoretical distribution to the histogram

fig = plt.figure(num=None, figsize=(750/96., 450/96.), dpi=96, facecolor='w', edgecolor=None)
ax = fig.add_subplot(111)



# frame = plt.gca()
ax.get_yaxis().set_visible(False)

start = gamma.ppf(0.0001, fit_alpha, loc=fit_loc, scale=fit_beta)
stop = gamma.ppf(0.999, fit_alpha, loc=fit_loc, scale=fit_beta)
x = np.linspace(start, stop, 100)
ax.plot(x, gamma.pdf(x, fit_alpha, loc=fit_loc, scale=fit_beta), 
        'r-', lw=5, alpha=0.6, label='Gamma Distribution')
ax.hist(data, normed=True, histtype='stepfilled', alpha=0.5)
ax.legend(loc='best', frameon=False)
plt.title('Fit of Gamma Distribution to IMR data')
plt.xlabel('Number of Deaths per 1000 Births')
plt.savefig('gamma_dist.jpg')
plt.show()

from scipy.stats import gamma, beta
from math import sqrt

def beta_params(m,s):
    '''convert mean and std to shape params for the beta distribution'''
    a = ((1-m)/(s**2) - 1/m)*m**2
    b = a*(1/m - 1)
    return a,b

def gamma_params(m,s):
    '''convert mean and std to shape params for the gamma distribution'''
    a = (m/s)**2
    b = (s**2)/m
    return a,b

def to_unity(values):
    '''convert a list of numbers to a list with the same proportions, but sums to unity'''
    s = 1.*sum(values)
    return [i/s for i in values]

# derive a theoretical gamma distribution from the U.S. IM rate data (this decade only)
rates, ranks = OAT(raw, 'none', how_much=0)

first_half = [str(i) for i in range(1995, 2005)]
second_half = [str(i) for i in range(2005, 2015)]
rates_first = rates[first_half]
rates_second = rates[second_half]

data = 1000*rates.values.flatten()

fit_alpha, fit_loc, fit_beta=gamma.fit(data)

# create the theoretical distribution by fitting to the data
us_gamma = gamma.rvs(fit_alpha, loc=fit_loc, scale=fit_beta, size=10000)

# compare the theoretical distribution to the histogram

fig = plt.figure(num=None, figsize=(750/96., 450/96.), dpi=96, facecolor='w', edgecolor=None)
ax = fig.add_subplot(111)


# frame = plt.gca()
ax.get_yaxis().set_visible(False)

start = gamma.ppf(0.0001, fit_alpha, loc=fit_loc, scale=fit_beta)
stop = gamma.ppf(0.999, fit_alpha, loc=fit_loc, scale=fit_beta)
x = np.linspace(start, stop, 100)
ax.plot(x, gamma.pdf(x, fit_alpha, loc=fit_loc, scale=fit_beta), 
        'r-', lw=5, alpha=0.6, label='Gamma Distribution')
ax.hist(data, normed=True, histtype='stepfilled', alpha=0.5)
ax.legend(loc='best', frameon=False)
plt.title('Fit of Gamma Distribution to IMR data')
plt.xlabel('Number of Deaths per 1000 Births')
plt.savefig('gamma_dist.jpg')
plt.show()

rates, ranks = OAT(raw, 'none', how_much=0)
rank_diffs = ranks.diff(axis=1)
rank_diffs['min_diff'] = rank_diffs.min(axis=1)
rank_diffs['max_diff'] = rank_diffs.max(axis=1)
rank_diffs['highest'] = ranks.min(axis=1)
rank_diffs['lowest'] = ranks.max(axis=1)
def hi_low_diff(row):
    hi = row.max() - row.min()
    return hi - low
rank_diffs['total_diff'] = ranks.apply(hi_low_diff, axis=1)
real_diffs = rank_diffs[['min_diff', 'max_diff', 'highest', 'lowest', 'total_diff']]
real_diffs[['total_diff']]

from random import choice
from string import ascii_uppercase

us_rv = gamma.rvs(fit_alpha, loc=fit_loc, scale=fit_beta, size=10000)
us_p = [gamma.cdf(i, fit_alpha, loc=fit_loc, scale=fit_beta) for i in us_rv]
us_p = to_unity(us_p)

records = []

for s in range(51):
    name = ''.join(choice(ascii_uppercase) for i in range(4))
    values = np.random.choice(us_rv, replace=True, size=10, p=us_p)
    row = [name]
    row.extend(values)
    records.append(row)
    
    
years = ['states']
years.extend(second_half)
test_1 = pd.DataFrame.from_records(records, columns=years, index='states')

ranks_1 = test_1.rank(axis=0)
ranks_1.diff(axis=1)

rank_diffs = ranks_1.diff(axis=1)
rank_diffs['min_diff'] = rank_diffs.min(axis=1)
rank_diffs['max_diff'] = rank_diffs.max(axis=1)
rank_diffs['highest'] = ranks_1.min(axis=1)
rank_diffs['lowest'] = ranks_1.max(axis=1)
def hi_low_diff(row):
    hi = row.max() - row.min()
    return hi - low
rank_diffs['total_diff'] = ranks_1.apply(hi_low_diff, axis=1)
simm_diffs = rank_diffs[['min_diff', 'max_diff', 'highest', 'lowest', 'total_diff']]
simm_diffs[['total_diff']]

# show the distribution of rank ranges for the simulated random data and the actual data

# simm_diffs
mask = real_diffs.index.isin(['United States'])
real_diffs_ = real_diffs[~mask]
real_ranges = real_diffs_[['total_diff']]

simm_ranges = simm_diffs[['total_diff']]

bins = range(0, 52, 3)

fig = plt.figure(num=None, figsize=(750/96., 450/96.), dpi=96, facecolor='w', edgecolor=None)
ax = fig.add_subplot(111)

ax.hist(real_ranges.values.flatten(), normed=False, histtype='stepfilled', 
        label='Actual data', alpha=0.5, color='b', bins=bins)

ax.hist(simm_ranges.values.flatten(), normed=False, histtype='stepfilled', 
        label='Simmulated data', alpha=0.5, color='g', bins=bins)

# fit_alpha, fit_loc, fit_beta=gamma.fit(data)

# # create the theoretical distribution by fitting to the data
# us_gamma = gamma.rvs(fit_alpha, loc=fit_loc, scale=fit_beta, size=10000)

# # compare the theoretical distribution to the histogram

# start = gamma.ppf(0.001, fit_alpha, loc=fit_loc, scale=fit_beta)
# stop = gamma.ppf(0.999, fit_alpha, loc=fit_loc, scale=fit_beta)
# x = np.linspace(start, stop, 100)
# ax.plot(x, gamma.pdf(x, fit_alpha, loc=fit_loc, scale=fit_beta), 
#         'r-', lw=5, alpha=0.6, label='theoretical dist.')

plt.annotate('This "bin" shows that 7 states \nhave a range between 21 and 24',(22.5, 7),(22, 12),va='center',ha='center', 
               rotation=0, arrowprops=dict(arrowstyle='->'), annotation_clip=False) 

plt.annotate('This "bin" shows that 13 states \nhave a range between 45 and 48',(46.5, 13),(42, 17),va='center',ha='center', 
               rotation=0, arrowprops=dict(arrowstyle='->'), annotation_clip=False) 

plt.annotate('ME',(43.5, 1),(43.5, 3),va='center',ha='center', 
               rotation=0, arrowprops=dict(arrowstyle='->'), annotation_clip=False) 

plt.annotate('SD',(46.5, 1),(46.5, 3),va='center',ha='center', 
               rotation=0, arrowprops=dict(arrowstyle='->'), annotation_clip=False) 

print bins
plt.xlim(-1, 53)
plt.ylim(0, 20)
plt.title('Rank Ranges of Simmulated vs. Actual Data\n(large range values mean the data bounce around a lot)')
plt.xlabel('Bins')
plt.ylabel('Number of States in the bin')
ax.legend(loc='upper left', frameon=False)
ax.set_xticks(bins)
plt.savefig('rank_ranges.jpg')
plt.show()

fig = plt.figure(num=None, figsize=(750/96., 450/96.), dpi=96, facecolor='w', edgecolor=None)
ax = fig.add_subplot(111)

x = range(1995, 2015)

births = df

import copy

births = copy.deepcopy(raw)
cols = births.columns
for col in cols:
    if 'Births' not in col:
        births.pop(col)

deaths = copy.deepcopy(raw)
cols = deaths.columns
for col in cols:
    if 'Deaths' not in col:
        deaths.pop(col)
        
maine_births = births[births.index.isin(['South Dakota'])]
maine_deaths = deaths[deaths.index.isin(['South Dakota'])]

first_half = ['{}_Deaths'.format(str(i)) for i in range(1995, 2005)]
second_half = ['{}_Deaths'.format(str(i)) for i in range(2005, 2015)]
_first = maine_deaths[first_half]
_second = maine_deaths[second_half]

print _first.values.flatten().sum(), _second.values.flatten().sum()


y = map(int, list(maine_births.values.flatten()))
lns1 = ax.plot(x, y, label='Births', color='g', linewidth=3)

ax2 = ax.twinx()
y = map(int, list(maine_deaths.values.flatten()))
lns2 = ax2.plot(x, y, label='Deaths', color='r', linewidth=3)

lns = lns1+lns2
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=0, frameon=False)

plt.xlim(1995,2014)
plt.title("Births and Infant Deaths in South Dakota")
ax.set_ylabel("Number of Births")
ax2.set_ylabel('Number of Deaths')

labels = x

plt.xticks(x, labels, rotation=45)
plt.setp( ax.xaxis.get_majorticklabels(), rotation=45 )
plt.savefig('south_dakota_births_deaths.jpg')
plt.show()

maine_deaths

ranks
