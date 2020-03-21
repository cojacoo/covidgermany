# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import pydeck
import glob
import statsmodels.api as sm
#import altair as alt
import plotly.graph_objects as go
import plotly.express as px

mapbox_access_token = 'pk.eyJ1IjoiY29qYWNrIiwiYSI6IkRTNjV1T2MifQ.EWzL4Qk-VvQoaeJBfE6VSA'
px.set_mapbox_access_token(mapbox_access_token)

st.title('COVID19 Germany')
det = st.checkbox('Text auf deutsch')
if det:
    st.markdown('Anhand von Landkreis-Daten der gemeldeten Infektionen mit Covid19 in Deutschland soll diese Anwendung dabei helfen, ein differenziertes Bild über die Entwicklungen an den unterschiedlichen Orten zu bekommen. Anders als Modell-Projektionen an anderen Stellen, werden hier tatsächliche Daten verwendet. Die Darstellungen und Werkzeuge sind dazu gedacht, anhand der tatsächlichen Entwicklungen die Debatte über angemessene Maßnahmen gegen die Ausbreitung des Virus anzuregen.')
else:
    st.markdown('Based on county-level data about Covid19 infections in Germany, this tool shall help to get a more nuanced image about the infection dynamics at different locations. Complementary to model projections elsewhere, the data visualisations and tools are intended to spark debates about balanced measures to counteract against the spread of the virus.')

@st.cache
def load_data():
    #get file list
    fi = glob.glob("./corona_*")
    
    #read all files into dataframe
    dummy = pd.read_csv(fi[0],index_col=0)
    dummy_cases = dummy.iloc[1:,:2].copy()
    dummy_kausali = dummy.iloc[1:,:2].copy()
    
    for fix in fi:
        dummy = pd.read_csv(fix,index_col=0)
        cases = pd.DataFrame(dummy.iloc[1:,2],dtype=float)
        cases.columns = [pd.to_datetime(dummy.iloc[1,-1])]
        dummy_cases = pd.concat([dummy_cases,cases],axis=1)
        
        if np.shape(dummy)[1]==5:
            kausali = pd.DataFrame(dummy.iloc[1:,3],dtype=float)
            kausali.columns = [pd.to_datetime(dummy.iloc[1,-1])]
            dummy_kausali = pd.concat([dummy_kausali,kausali],axis=1)

    LKx = pd.read_csv('LKposi.csv',index_col=0)
    dummy_cases = pd.concat([LKx,dummy_cases[dummy_cases.columns[2:].sort_values()]],axis=1)
    dummy_kausali = pd.concat([LKx,dummy_kausali[dummy_kausali.columns[2:].sort_values()]],axis=1)

    dummy_casesx = dummy_cases.iloc[:,5:].T
    dummy_casesx.index = pd.to_datetime(dummy_casesx.index)
    
    dummy_casesx2 = dummy_casesx.resample('1D').max().T
    dummy_increase = dummy_casesx.resample('1D').max().T.diff(2,1)
    
    dummy_frowfac = dummy_increase.copy()
    for i in np.arange(len(dummy_frowfac.columns))[1:][::-1]:
        dummy_frowfac.iloc[:,i] = dummy_frowfac.iloc[:,i].div(dummy_frowfac.iloc[:,i-1])

    return [pd.concat([LKx,dummy_casesx2],axis=1),dummy_increase,dummy_frowfac,LKx]

data_load_state = st.text('Loading data...')
[data_case,data_increase,data_frowfac,LKx] = load_data()
data_load_state.text('Loading data... done!')

if det:
    st.subheader('Falldynamik in Deutschland')
    st.markdown('Für einen Überblick über die Ausbreitung des Virus in Deutschland gibt es in diesem Bereich Kartendarstellungen der Landkreis-Daten. Da das Virus von Mensch zu Mensch übertragen wird und die Landkreise sehr unterschiedlich groß sind, gibt es neben den absoluten Fallzahlen (Cases) auch relative Fallzahlen bezogen auf 100.000 Einwohner*innen. Mit dem Schieberegler, kann der angezeigte Tag gesteuert werden, um ein Gefühl für die Gesamtentwicklung zu erhalten.')
    st.markdown('Neben der Darstellung mit Punktmarkierungen der Landkreise, kann die Karte auch als Auftretensverteilung angezeigt werden (heatmap). Für die zeitliche Veränderung gibt es darin auch den täglichen Anstieg (increase) sowie das Verhältnis des Wachstums vom Tag zum Vortag (increase ratio) abgebildet.')
else:
    st.subheader('Case dynamics in Germany')
    st.markdown('To get an overview about the general speading of the virus across Germany, this section reports different maps based on the county-level case data. Since the virus transmits from person to person and since the counties have very different numbers of inhabitants, we have added relative to capita data. The slider allows to control the plotted day to get a feeling about the general dynamics.')
    st.markdown('In addition to the dotted plot of infection cases, there is also a nation-wide heatmap. To elaborate a little more on the temporal dynamics, daily increase and the increase ratio to the previous day is reported.')

data_sel = st.selectbox('Select Data',['Cases','Cases per 100000 capita','Case increase','Case increase per 100000 capita','Increase ratio'],0)
if data_sel=='Cases':
    data_cases = data_case
elif data_sel=='Cases per 100000 capita':
    data_cases = pd.concat([LKx,data_case.iloc[:,5:].div(LKx.EWZ,axis=0)*100000.],axis=1)
elif data_sel=='Case increase':
    data_cases = pd.concat([LKx,data_increase],axis=1)
elif data_sel=='Case increase per 100000 capita':
    data_cases = pd.concat([LKx,data_increase.div(LKx.EWZ,axis=0)*100000.],axis=1)
elif data_sel=='Increase ratio':
    data_cases = pd.concat([LKx,data_frowfac],axis=1)


if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data_cases)

#helper function
#jiter cases
dlat = 0.6#np.exp(np.log(data_cases.lat.sort_values().diff()).median())*50.
dlon = 0.6#np.exp(np.log(data_cases.lon.sort_values().diff()).median())*50.

def jiter_data(data,co):
    firstitem = True
    for i in data.index:
        n = int(data_cases.loc[i,co])
        c = pd.DataFrame((np.random.randn(2*n).reshape((n,2))*0.5)*np.array([dlat,dlon])+data.loc[i,['lat','lon']].values,columns=['lat','lon'])
        if firstitem:
            dummyj = c
            firstitem = False
        else:
            dummyj = pd.concat([dummyj,c])
    dummyj = dummyj.reset_index()
    return dummyj

hmplot = st.checkbox('Show heatmap')
di = st.slider('day', 4, len(data_cases.columns)-1, len(data_cases.columns)-1)
dranx = len(data_cases.columns)-1
datex = data_cases.columns[di]
    
if ((data_sel=='Cases') | (data_sel=='Cases per 100000 capita')) & (hmplot==True):
    #show map with hexagon heatmap
    
    # Define a layer to display on a map
    layer = pydeck.Layer(
       'HexagonLayer',
       jiter_data(data_cases,datex)[['lon', 'lat']],
       get_position=['lon', 'lat'],
       auto_highlight=True,
       elevation_scale=100,
       pickable=True,
       colorRange=[[69,2,86],[59,28,140],[33,144,141],[90,200,101],[249,231,33]],
       elevation_range=[0,3000],
       elevationDomain=[0,12],
       extruded=True,
       coverage=10)
    
    # Set the viewport location
    view_state = pydeck.ViewState(
        longitude=8.815,
        latitude=51.155323,
        zoom=5,
        pitch=25.5,
        bearing=0.)
    
    st.subheader('Map of COVID-19 cases on ' + str(datex.date()))
    st.pydeck_chart(
        pydeck.Deck(layers=[layer], initial_view_state=view_state, mapbox_key='pk.eyJ1IjoiY29qYWNrIiwiYSI6IkRTNjV1T2MifQ.EWzL4Qk-VvQoaeJBfE6VSA')
        )

elif (data_sel=='Cases') & (hmplot==False):
    data_cases1 = pd.concat([data_cases[['lat','lon',datex,'Landkreis','EWZ']],data_increase[datex],data_frowfac[datex]],axis=1)
    data_cases1.columns = ['lat','lon','cases','Landkreis','capita','increase','growth']
    fig = px.scatter_mapbox(data_cases1, lat="lat", lon="lon",  size='cases', color=np.log10(data_cases1.cases),
                  hover_name='Landkreis',hover_data=['capita','cases','increase','growth'],range_color=[0.,np.log10(data_cases.iloc[:,-1].quantile(0.99))],
                  color_continuous_scale=px.colors.sequential.Cividis, size_max=24, zoom=5, height=600)
    fig.update_layout(coloraxis_colorbar=dict(
        title="Cases",
        tickvals=[0,1,2,3],
        ticktext=['1' , '10', '100', '1000'],
        ))
    st.subheader('Map of COVID-19 cases on ' + str(datex.date()))
    st.plotly_chart(fig)

elif (data_sel=='Cases per 100000 capita') & (hmplot==False):
    data_cases1 = pd.concat([data_cases[['lat','lon',datex,'Landkreis','EWZ']],data_increase[datex],data_frowfac[datex]],axis=1)
    data_cases1.columns = ['lat','lon','cases','Landkreis','capita','increase','growth']
    fig = px.scatter_mapbox(data_cases1, lat="lat", lon="lon",  size='cases', color='cases',
                  hover_name='Landkreis',hover_data=['capita','cases','increase','growth'],range_color=[0.,data_cases.iloc[:,-1].quantile(0.99)],
                  color_continuous_scale=px.colors.sequential.Cividis, size_max=24, zoom=5, height=600)
    st.subheader('Map of COVID-19 cases on ' + str(datex.date()))
    st.plotly_chart(fig)

else:
    data_cases1 = pd.concat([data_case[['lat','lon',datex,'Landkreis','EWZ']],data_case[datex].div(LKx.EWZ,axis=0)*100000.,data_increase[datex],data_increase[datex].div(LKx.EWZ,axis=0)*100000.,data_frowfac[datex]],axis=1)
    data_cases1.columns = ['lat','lon','cases','Landkreis','capita','cases per 100000','increase','increase per 100000','growth']
    if (data_sel=='Case increase'):
        fig = px.scatter_mapbox(data_cases1, lat="lat", lon="lon",  size='cases per 100000', color='increase',
                  hover_name='Landkreis',hover_data=['capita','cases','cases per 100000','increase','increase per 100000','growth'],range_color=[data_increase.quantile(0.95).quantile(0.95)*-1.,data_increase.quantile(0.95).quantile(0.95)],
                  color_continuous_scale=px.colors.diverging.Portland, size_max=24, zoom=5, height=600)
        st.subheader('Map of COVID-19 case increase on ' + str(datex.date()))
        st.plotly_chart(fig)
    elif (data_sel=='Case increase per 100000 capita'):
        fig = px.scatter_mapbox(data_cases1, lat="lat", lon="lon",  size='cases per 100000', color='increase per 100000',
                  hover_name='Landkreis',hover_data=['capita','cases','cases per 100000','increase','increase per 100000','growth'],range_color=[(data_increase.div(LKx.EWZ,axis=0)*100000.).quantile(0.95).quantile(0.95)*-1.,(data_increase.div(LKx.EWZ,axis=0)*100000.).quantile(0.95).quantile(0.95)],
                  color_continuous_scale=px.colors.diverging.Portland, size_max=24, zoom=5, height=600)
        st.subheader('Map of COVID-19 case increase per 100000 on ' + str(datex.date()))
        st.plotly_chart(fig)
    elif (data_sel=='Increase ratio'):
        fig = px.scatter_mapbox(data_cases1, lat="lat", lon="lon",  size='cases per 100000', color='growth',
                  hover_name='Landkreis',hover_data=['capita','cases','cases per 100000','increase','increase per 100000','growth'],range_color=[0.3,1.7],
                  color_continuous_scale=px.colors.diverging.Portland, size_max=24, zoom=5, height=600)
        st.subheader('Map of COVID-19 case growth on ' + str(datex.date()))
        st.plotly_chart(fig)
    


if det:
    st.subheader('Entwicklung der Fallzahlen und exponentielles/logistisches Wachstum')
    st.markdown('Für einen genaueren Blick in eine Region (Bundesland oder Landkreis) stellen wir nun die zeitliche Entwicklung der Fallzahlen dar und passen drei Modelle an. Das erste exponentielle Modell nutzt alle Daten für die automatische Anpassung. Das zweite exponentielle Modell nutzt nur die letzten Tage für die Anpassung und erlaubt damit eine schnelle Inspektion, ob die letzten Entwicklungen einer anderen Wachstumskurve folgen. Das dritte Modell ist das logistische - also gesättigte - Wachstum. Bei letzterem muss allerdings eine Annahme über die maximalen Fallzahlen getroffen werden. Wenn wir mit den Maßnahmen zur räumlichen Abgrenzung erfolgreich sind, sollten die Beobachtungen mehr und mehr von der exponentiellen Entwicklung abweichen und immer besser von einer logistischen Funktion abgebildet werden können. Auch hier können wieder absolute Fallzahlen oder Fälle pro 100.000 Einwohner*innen angezeigt werden.')
    st.markdown('Ein weiterer Regler erlaubt es, die Zeitreihe länger oder kürzer zu zeigen.')
else:
    st.subheader('County-level data and exponential/logistic growth')
    st.markdown('For a closer look into a region (state or county) the following will present the respective development of cases and fits three models. The first exponential model uses all available data for automated regression. The second will only fit to the last days. Hence a deviation between the two models allows for a quick inspection if the time segments follow different growth curves. The third model is the logistic (limited) growth model. For the latter an assumption about the maximum of cases has to be provided (controlled by a scaling factor of the current maximum). If the current measures of spatial distancing are successful, the observed cases should deviate more and more from the exponential model towards a better fit with the logistic one. Again, you can select to show absolute numbers or cases per 100,000 capita.')
    st.markdown('A further slider controls the length of the shown time series.')

BL = st.selectbox('Select Bundesland',np.append(np.array('Alle'),LKx.Bundesland.unique()),0)
LK = st.selectbox('Select Landkreis',np.append(np.array('Alle'),LKx.loc[LKx.Bundesland==BL,'Landkreis'].unique()),0)

scmax = st.slider('Saturation at x times of the current maximum (for logistic model)', 1., 100., 10.)
dran = st.slider('no. of days shown', 8, 200, int(np.round(dranx*1.3)))

percap = st.checkbox('Cases per 100000 capita')
if percap:
    data_cases = pd.concat([LKx,data_case.iloc[:,5:].div(LKx.EWZ,axis=0)*100000.],axis=1)
else:
    data_cases = data_case

if BL=='Alle':
    if percap:
        dc = pd.DataFrame(data_cases.mean(axis=0).iloc[5:].drop_duplicates('first'))
    else:
        dc = pd.DataFrame(data_cases.sum(axis=0).iloc[5:].drop_duplicates('first'))
    dc.columns = ['Germany']
elif LK=='Alle':
    if percap:
        dc = pd.DataFrame(data_cases.loc[data_cases.Bundesland==BL].mean(axis=0).iloc[5:].drop_duplicates('first'))
    else:
        dc = pd.DataFrame(data_cases.loc[data_cases.Bundesland==BL].sum(axis=0).iloc[5:].drop_duplicates('first'))
    dc.columns = [BL]
else:
    dc = data_cases.loc[(data_cases.Bundesland==BL) & (data_cases.Landkreis==LK)].iloc[:,5:].T.drop_duplicates()
    dc.columns = [LK]
    dc.index = pd.to_datetime(dc.index)

if st.checkbox('Show record data'):
    st.write(dc)

linscaley = st.checkbox('Linear y-axis of cases')
st.markdown('(note that the y-axis is logarithmic on default since the process of infection spreading is exponential)')

X = ((dc.index-dc.index[0]).days + (dc.index-dc.index[0]).seconds/86400.).values
X = sm.add_constant(X)

y = np.log(dc.iloc[:,0].values.astype(float))
dcmx = dc.iloc[:,0].max()
yt = dc.iloc[:,0].values.astype(float)/(dcmx*scmax)

#exponential model
lm = sm.OLS(y,X)
res = lm.fit()

#logistic model
lo = sm.Logit(yt,X)
resl = lo.fit()

#exponential model for only last days
lm1 = sm.OLS(np.append(y[0],y[-6:]),np.append(X[0],X[-6:]).reshape(7,2))
res1 = lm1.fit()
#res1.params


fig = go.Figure()
fig.add_trace(go.Scatter(x=dc.index,y=dc[dc.columns[0]],mode='markers', name=dc.columns[0]))
fig.add_trace(go.Scatter(x=dc.index[0]+pd.to_timedelta(np.arange(dran*10)/10., unit='d'), y=np.exp(res.predict(sm.add_constant(np.arange(dran*10)/10.))),line=dict(dash='dash', width=1),name='Exponential Model'))
fig.add_trace(go.Scatter(x=dc.index[0]+pd.to_timedelta(np.arange(dran*10)/10., unit='d'), y=np.exp(res1.predict(sm.add_constant(np.arange(dran*10)/10.))),line=dict(dash='dash', width=1),name='Exponential Model (fitted to last days)'))
fig.add_trace(go.Scatter(x=dc.index[0]+pd.to_timedelta(np.arange(dran*10)/10., unit='d'), y=resl.predict(sm.add_constant(np.arange(dran*10)/10.))*(dcmx*scmax),line=dict(dash='dash', width=1),name='Logistic Model'))

if linscaley:
    yaxtype='linear'
else:
    yaxtype='log'

fig.update_layout(
    xaxis=dict(
        showline=True,
        showgrid=True,
        showticklabels=True,
        linecolor='rgb(204, 204, 204)',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(82, 82, 82)',
        ),
    ),
    yaxis=dict(
        showline=True,
        showgrid=True,
        showticklabels=True,
        linecolor='rgb(204, 204, 204)',
        linewidth=2,
        type=yaxtype,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(82, 82, 82)',
        ),
    ),
    autosize=True,
    margin=dict(
        autoexpand=True,
        l=100,
        r=20,
        t=110,
    ),
    showlegend=True,
    template='plotly_white',
    plot_bgcolor='white',
    legend=dict(x=0.02, y=0.98)
)

st.plotly_chart(fig)

st.markdown('All data is without any liability and originates from presentations of regional authorities.')
st.markdown('(cc) Conrad Jackisch. Please join, fork, reuse: https://github.com/cojacoo/covidgermany')

