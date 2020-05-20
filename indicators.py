import numpy as np
import pandas as pd
import copy

# read csv file

data = pd.read_csv("IBM.csv")
data.rename(columns={'timestamp': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close','volume': 'Volume'}, inplace=True)
TechIndicator = copy.deepcopy(data)

# Relative Strength Index
# Avg(PriceUp)/(Avg(PriceUP)+Avg(PriceDown)*100
# Where: PriceUp(t)=1*(Price(t)-Price(t-1)){Price(t)- Price(t-1)>0};
#        PriceDown(t)=-1*(Price(t)-Price(t-1)){Price(t)- Price(t-1)<0};


def rsi(values):
    up = values[values>0].mean()
    down = -1*values[values<0].mean()
    return 100 * up / (up + down)


# Add Momentum_1D column for all 15 stocks.
# Momentum_1D = P(t) - P(t-1)

TechIndicator['Momentum_1D'] = (TechIndicator['Close']-TechIndicator['Close'].shift(1)).fillna(0)
TechIndicator['RSI_14D'] = TechIndicator['Momentum_1D'].rolling(center=False, window=14).apply(rsi).fillna(0)

#  Calculation of Volume (Plain)

TechIndicator['Volume_plain'] = TechIndicator['Volume'].fillna(0)

#  Calculation of Bolinger Band


def bbands(price, length=30, numsd=2):
    """ returns average, upper band, and lower band"""
    #  ave = pd.stats.moments.rolling_mean(price,length)
    ave = price.rolling(window=length, center=False).mean()
    #  sd = pd.stats.moments.rolling_std(price,length)
    sd = price.rolling(window=length, center=False).std()
    upband = ave + (sd*numsd)
    dnband = ave - (sd*numsd)
    return np.round(ave, 3), np.round(upband, 3), np.round(dnband, 3)


TechIndicator['BB_Middle_Band'], TechIndicator['BB_Upper_Band'], TechIndicator['BB_Lower_Band'] = bbands(TechIndicator['Close'], length=20, numsd=1)
TechIndicator['BB_Middle_Band'] = TechIndicator['BB_Middle_Band'].fillna(0)
TechIndicator['BB_Upper_Band'] = TechIndicator['BB_Upper_Band'].fillna(0)
TechIndicator['BB_Lower_Band'] = TechIndicator['BB_Lower_Band'].fillna(0)

#  Calculation of Aroon


def aroon(df, tf=25):
    aroonup = []
    aroondown = []
    x = tf
    while x < len(df['Date']):
        aroon_up = ((df['High'][x-tf:x].tolist().index(max(df['High'][x-tf:x])))/float(tf))*100
        aroon_down = ((df['Low'][x-tf:x].tolist().index(min(df['Low'][x-tf:x])))/float(tf))*100
        aroonup.append(aroon_up)
        aroondown.append(aroon_down)
        x += 1
    return aroonup, aroondown


listofzeros = [0] * 25
up, down = aroon(TechIndicator)
aroon_list = [x - y for x, y in zip(up, down)]
if len(aroon_list) == 0:
    aroon_list = [0] * TechIndicator.shape[0]
    TechIndicator['Aroon_Oscillator'] = aroon_list
else:
    TechIndicator['Aroon_Oscillator'] = listofzeros + aroon_list


#  Calculation of Price Volume Trend
#  PVT = [((CurrentClose - PreviousClose) / PreviousClose) x Volume] + PreviousPVT

TechIndicator["PVT"] = (TechIndicator['Momentum_1D']/ TechIndicator['Close'].shift(1))*TechIndicator['Volume']
TechIndicator["PVT"] = TechIndicator["PVT"]-TechIndicator["PVT"].shift(1)
TechIndicator["PVT"] = TechIndicator["PVT"].fillna(0)

#  Calculation of Acceleration Bands


def abands(df):
    #  df['AB_Middle_Band'] = pd.rolling_mean(df['Close'], 20)
    df['AB_Middle_Band'] = df['Close'].rolling(window = 20, center=False).mean()
    # High * ( 1 + 4 * (High - Low) / (High + Low))
    df['aupband'] = df['High'] * (1 + 4 * (df['High']-df['Low'])/(df['High']+df['Low']))
    df['AB_Upper_Band'] = df['aupband'].rolling(window=20, center=False).mean()
    # Low *(1 - 4 * (High - Low)/ (High + Low))
    df['adownband'] = df['Low'] * (1 - 4 * (df['High']-df['Low'])/(df['High']+df['Low']))
    df['AB_Lower_Band'] = df['adownband'].rolling(window=20, center=False).mean()


abands(TechIndicator)
TechIndicator = TechIndicator.fillna(0)

#  Drop unwanted columns

columns2Drop = ['Momentum_1D', 'aupband', 'adownband']
TechIndicator = TechIndicator.drop(labels=columns2Drop, axis=1)


#  Calculation of Stochastic Oscillator (%K and %D)


def STOK(df, n):
    df['STOK'] = ((df['Close'] - df['Low'].rolling(window=n, center=False).mean()) / (df['High'].rolling(window=n, center=False).max() - df['Low'].rolling(window=n, center=False).min())) * 100
    df['STOD'] = df['STOK'].rolling(window=3, center=False).mean()


STOK(TechIndicator, 4)
TechIndicator = TechIndicator.fillna(0)


#  Calculation of Chaikin Money Flow


def CMFlow(df, tf):
    CHMF = []
    MFMs = []
    MFVs = []
    x = tf

    while x < len(df['Date']):
        PeriodVolume = 0
        volRange = df['Volume'][x - tf:x]
        for eachVol in volRange:
            PeriodVolume += eachVol

        MFM = ((df['Close'][x] - df['Low'][x]) - (df['High'][x] - df['Close'][x])) / (df['High'][x] - df['Low'][x])
        MFV = MFM * PeriodVolume

        MFMs.append(MFM)
        MFVs.append(MFV)
        x += 1

    y = tf
    while y < len(MFVs):
        PeriodVolume = 0
        volRange = df['Volume'][x - tf:x]
        for eachVol in volRange:
            PeriodVolume += eachVol
        consider = MFVs[y - tf:y]
        tfsMFV = 0

        for eachMFV in consider:
            tfsMFV += eachMFV

        tfsCMF = tfsMFV / PeriodVolume
        CHMF.append(tfsCMF)
        y += 1
    return CHMF


listofzeros = [0] * 40
CHMF = CMFlow(TechIndicator, 20)
if len(CHMF) == 0:
    CHMF = [0] * TechIndicator.shape[0]
    TechIndicator['Chaikin_MF'] = CHMF
else:
    TechIndicator['Chaikin_MF'] = listofzeros + CHMF


# Calculation of Parabolic SAR


def psar(df, iaf = 0.02, maxaf = 0.2):
    length = len(df)
    dates = (df['Date'])
    high = (df['High'])
    low = (df['Low'])
    close = (df['Close'])
    psar = df['Close'][0:len(df['Close'])]
    psarbull = [None] * length
    psarbear = [None] * length
    bull = True
    af = iaf
    ep = df['Low'][0]
    hp = df['High'][0]
    lp = df['Low'][0]
    for i in range(2,length):
        if bull:
            psar[i] = psar[i - 1] + af * (hp - psar[i - 1])
        else:
            psar[i] = psar[i - 1] + af * (lp - psar[i - 1])
        reverse = False
        if bull:
            if df['Low'][i] < psar[i]:
                bull = False
                reverse = True
                psar[i] = hp
                lp = df['Low'][i]
                af = iaf
        else:
            if df['High'][i] > psar[i]:
                bull = True
                reverse = True
                psar[i] = lp
                hp = df['High'][i]
                af = iaf
        if not reverse:
            if bull:
                if df['High'][i] > hp:
                    hp = df['High'][i]
                    af = min(af + iaf, maxaf)
                if df['Low'][i - 1] < psar[i]:
                    psar[i] = df['Low'][i - 1]
                if df['Low'][i - 2] < psar[i]:
                    psar[i] = df['Low'][i - 2]
            else:
                if df['Low'][i] < lp:
                    lp = df['Low'][i]
                    af = min(af + iaf, maxaf)
                if df['High'][i - 1] > psar[i]:
                    psar[i] = df['High'][i - 1]
                if df['High'][i - 2] > psar[i]:
                    psar[i] = df['High'][i - 2]
        if bull:
            psarbull[i] = psar[i]
        else:
            psarbear[i] = psar[i]
    df['psar'] = psar


psar(TechIndicator)


# Calculation of Price Rate of Change


TechIndicator['ROC'] = ((TechIndicator['Close'] - TechIndicator['Close'].shift(12))/(TechIndicator['Close'].shift(12)))*100
TechIndicator = TechIndicator.fillna(0)

# Calculation of Volume Weighted Average Price


TechIndicator['VWAP'] = np.cumsum(TechIndicator['Volume'] * (TechIndicator['High'] + TechIndicator['Low'])/2) / np.cumsum(TechIndicator['Volume'])
TechIndicator = TechIndicator.fillna(0)


# Calculation of Momentum


TechIndicator['Momentum'] = TechIndicator['Close'] - TechIndicator['Close'].shift(4)
TechIndicator = TechIndicator.fillna(0)


# Calculation of Commodity Channel Index


def CCI(df, n, constant):
    TP = (df['High'] + df['Low'] + df['Close']) / 3
    CCI = pd.Series((TP - TP.rolling(window=n, center=False).mean()) / (constant * TP.rolling(window=n, center=False).std())) #, name = 'CCI_' + str(n))
    return CCI


TechIndicator['CCI'] = CCI(TechIndicator, 20, 0.015)
TechIndicator = TechIndicator.fillna(0)


# Calculation of On Balance Volume


'''If the closing price is above the prior close price then: Current OBV = Previous OBV + Current Volume
If the closing price is below the prior close price then: Current OBV = Previous OBV - Current Volume
If the closing prices equals the prior close price then: Current OBV = Previous OBV (no change)'''


new = (TechIndicator['Volume'] * (~TechIndicator['Close'].diff().le(0) * 2 -1)).cumsum()
TechIndicator['OBV'] = new


# Calcualtion of Keltner Channels


def KELCH(df, n):
    KelChM = pd.Series(((df['High'] + df['Low'] + df['Close']) / 3).rolling(window =n, center=False).mean(), name = 'KelChM_' + str(n))
    KelChU = pd.Series(((4 * df['High'] - 2 * df['Low'] + df['Close']) / 3).rolling(window =n, center=False).mean(), name = 'KelChU_' + str(n))
    KelChD = pd.Series(((-2 * df['High'] + 4 * df['Low'] + df['Close']) / 3).rolling(window =n, center=False).mean(), name = 'KelChD_' + str(n))
    return KelChM, KelChD, KelChU


KelchM, KelchD, KelchU = KELCH(TechIndicator, 14)
TechIndicator['Kelch_Upper'] = KelchU
TechIndicator['Kelch_Middle'] = KelchM
TechIndicator['Kelch_Down'] = KelchD
TechIndicator = TechIndicator.fillna(0)


# Calculation of Triple Exponential Moving Average
'''Triple Exponential MA Formula:

T-EMA = (3EMA – 3EMA(EMA)) + EMA(EMA(EMA))

Where:

EMA = EMA(1) + α * (Close – EMA(1))

α = 2 / (N + 1)

N = The smoothing period.'''

TechIndicator['EMA'] = TechIndicator['Close'].ewm(span=3, min_periods=0, adjust=True, ignore_na=False).mean()
TechIndicator = TechIndicator.fillna(0)
TechIndicator['TEMA'] = (3 * TechIndicator['EMA'] - 3 * TechIndicator['EMA'] * TechIndicator['EMA']) + (TechIndicator['EMA']*TechIndicator['EMA']*TechIndicator['EMA'])


# Calculation of Normalized Average True Range
'''True Range = Highest of (HIgh - low, abs(High - previous close), abs(low - previous close))

Average True Range = 14 day MA of True Range

Normalized Average True Range = ATR / Close * 100'''


TechIndicator['HL'] = TechIndicator['High'] - TechIndicator['Low']
TechIndicator['absHC'] = abs(TechIndicator['High'] - TechIndicator['Close'].shift(1))
TechIndicator['absLC'] = abs(TechIndicator['Low'] - TechIndicator['Close'].shift(1))
TechIndicator['TR'] = TechIndicator[['HL', 'absHC', 'absLC']].max(axis=1)
TechIndicator['ATR'] = TechIndicator['TR'].rolling(window=14).mean()
TechIndicator['NATR'] = (TechIndicator['ATR'] / TechIndicator['Close']) * 100
TechIndicator = TechIndicator.fillna(0)


# Calculation of Average Directional Movement Index (ADX)


def DMI(df, period):
    df['UpMove'] = df['High'] - df['High'].shift(1)
    df['DownMove'] = df['Low'].shift(1) - df['Low']
    df['Zero'] = 0

    df['PlusDM'] = np.where((df['UpMove'] > df['DownMove']) & (df['UpMove'] > df['Zero']), df['UpMove'], 0)
    df['MinusDM'] = np.where((df['UpMove'] < df['DownMove']) & (df['DownMove'] > df['Zero']), df['DownMove'], 0)

    df['plusDI'] = 100 * (df['PlusDM']/df['ATR']).ewm(span=period,min_periods=0,adjust=True,ignore_na=False).mean()
    df['minusDI'] = 100 * (df['MinusDM']/df['ATR']).ewm(span=period,min_periods=0,adjust=True,ignore_na=False).mean()

    df['ADX'] = 100 * (abs((df['plusDI'] - df['minusDI'])/(df['plusDI'] + df['minusDI']))).ewm(span=period,min_periods=0,adjust=True,ignore_na=False).mean()


DMI(TechIndicator, 14)
TechIndicator = TechIndicator.fillna(0)


# Drop Unwanted Columns
columns2Drop = ['UpMove', 'DownMove', 'ATR', 'PlusDM', 'MinusDM', 'Zero', 'EMA', 'HL', 'absHC', 'absLC', 'TR']
TechIndicator = TechIndicator.drop(labels=columns2Drop, axis=1)


# Calculation of MACD


TechIndicator['26_ema'] = TechIndicator['Close'].ewm(span=26,min_periods=0,adjust=True,ignore_na=False).mean()
TechIndicator['12_ema'] = TechIndicator['Close'].ewm(span=12,min_periods=0,adjust=True,ignore_na=False).mean()
TechIndicator['MACD'] = TechIndicator['12_ema'] - TechIndicator['26_ema']
TechIndicator = TechIndicator.fillna(0)


# Calculation of Money Flow Index


def MFI(df):
    # typical price
    df['tp'] = (df['High'] + df['Low'] + df['Close']) / 3
    # raw money flow
    df['rmf'] = df['tp'] * df['Volume']

    # positive and negative money flow
    df['pmf'] = np.where(df['tp'] > df['tp'].shift(1), df['tp'], 0)
    df['nmf'] = np.where(df['tp'] < df['tp'].shift(1), df['tp'], 0)

    # money flow ratio
    df['mfr'] = df['pmf'].rolling(window=14, center=False).sum() / df['nmf'].rolling(window=14, center=False).sum()
    df['Money_Flow_Index'] = 100 - 100 / (1 + df['mfr'])


MFI(TechIndicator)
TechIndicator = TechIndicator.fillna(0)


# Calculations of Ichimoku Cloud


def ichimoku(df):
    # Turning Line
    period9_high = df['High'].rolling(window=9, center=False).max()
    period9_low = df['Low'].rolling(window=9, center=False).min()
    df['turning_line'] = (period9_high + period9_low) / 2

    # Standard Line
    period26_high = df['High'].rolling(window=26, center=False).max()
    period26_low = df['Low'].rolling(window=26, center=False).min()
    df['standard_line'] = (period26_high + period26_low) / 2

    # Leading Span 1
    df['ichimoku_span1'] = ((df['turning_line'] + df['standard_line']) / 2).shift(26)

    # Leading Span 2
    period52_high = df['High'].rolling(window=52, center=False).max()
    period52_low = df['Low'].rolling(window=52, center=False).min()
    df['ichimoku_span2'] = ((period52_high + period52_low) / 2).shift(26)

    # The most current closing price plotted 22 time periods behind (optional)
    df['chikou_span'] = df['Close'].shift(-22)  # 22 according to investopedia


ichimoku(TechIndicator)
TechIndicator = TechIndicator.fillna(0)

# Calculation of William %R


def WillR(df):
    highest_high = df['High'].rolling(window=14, center=False).max()
    lowest_low = df['Low'].rolling(window=14, center=False).min()
    df['WillR'] = (-100) * ((highest_high - df['Close']) / (highest_high - lowest_low))


WillR(TechIndicator)
TechIndicator = TechIndicator.fillna(0)


# Calculation of MINMAX


def MINMAX(df):
    df['MIN_Volume'] = df['Volume'].rolling(window=14, center=False).min()
    df['MAX_Volume'] = df['Volume'].rolling(window=14, center=False).max()


MINMAX(TechIndicator)
TechIndicator = TechIndicator.fillna(0)


# Calculation of Adaptive Moving Average


def KAMA(price, n=10, pow1=2, pow2=30):
    ''' kama indicator '''
    ''' accepts pandas dataframe of prices '''

    absDiffx = abs(price - price.shift(1))

    ER_num = abs(price - price.shift(n))
    ER_den = absDiffx.rolling(window=n, center=False).sum()
    ER = ER_num / ER_den

    sc = (ER*(2.0/(pow1+1)-2.0/(pow2+1.0))+2/(pow2+1.0)) ** 2.0

    answer = np.zeros(sc.size)
    N = len(answer)
    first_value = True

    for i in range(N):
        if sc[i] != sc[i]:
            answer[i] = np.nan
        else:
            if first_value:
                answer[i] = price[i]
                first_value = False
            else:
                answer[i] = answer[i-1] + sc[i] * (price[i] - answer[i-1])
    return answer


TechIndicator['KAMA'] = KAMA(TechIndicator['Close'])
TechIndicator = TechIndicator.fillna(0)


# Drop Unwanted Columns

columns2Drop = ['26_ema', '12_ema', 'tp', 'rmf', 'pmf', 'nmf', 'mfr']
TechIndicator = TechIndicator.drop(labels=columns2Drop, axis=1)

TechIndicator.index = TechIndicator['Date']
TechIndicator = TechIndicator.drop(labels=['Date'], axis=1)
print(TechIndicator.tail())