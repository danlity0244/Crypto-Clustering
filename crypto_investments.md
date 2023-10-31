# Module 10 Application

## Challenge: Crypto Clustering

In this Challenge, you’ll combine your financial Python programming skills with the new unsupervised learning skills that you acquired in this module.

The CSV file provided for this challenge contains price change data of cryptocurrencies in different periods.

The steps for this challenge are broken out into the following sections:

* Import the Data (provided in the starter code)
* Prepare the Data (provided in the starter code)
* Find the Best Value for `k` Using the Original Data
* Cluster Cryptocurrencies with K-means Using the Original Data
* Optimize Clusters with Principal Component Analysis
* Find the Best Value for `k` Using the PCA Data
* Cluster the Cryptocurrencies with K-means Using the PCA Data
* Visualize and Compare the Results

### Import the Data

This section imports the data into a new DataFrame. It follows these steps:

1. Read  the “crypto_market_data.csv” file from the Resources folder into a DataFrame, and use `index_col="coin_id"` to set the cryptocurrency name as the index. Review the DataFrame.

2. Generate the summary statistics, and use HvPlot to visualize your data to observe what your DataFrame contains.


> **Rewind:** The [Pandas`describe()`function](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.describe.html) generates summary statistics for a DataFrame. 


```python
# Import required libraries and dependencies
import pandas as pd
import hvplot.pandas
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
```


```python
# Load the data into a Pandas DataFrame
df_market_data = pd.read_csv(
    Path(r"C:\Users\HPs\Desktop\BootCamp\Module 10\Activities\UTOR-VIRT-FIN-PT-08-2023-U-LOLC-main\10-Unsupervised-Learning\Unit 10 Homework\Instructions\Starter_Code\Resources\crypto_market_data.csv"),
    index_col="coin_id")

# Display sample data
df_market_data.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price_change_percentage_24h</th>
      <th>price_change_percentage_7d</th>
      <th>price_change_percentage_14d</th>
      <th>price_change_percentage_30d</th>
      <th>price_change_percentage_60d</th>
      <th>price_change_percentage_200d</th>
      <th>price_change_percentage_1y</th>
    </tr>
    <tr>
      <th>coin_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>bitcoin</th>
      <td>1.08388</td>
      <td>7.60278</td>
      <td>6.57509</td>
      <td>7.67258</td>
      <td>-3.25185</td>
      <td>83.51840</td>
      <td>37.51761</td>
    </tr>
    <tr>
      <th>ethereum</th>
      <td>0.22392</td>
      <td>10.38134</td>
      <td>4.80849</td>
      <td>0.13169</td>
      <td>-12.88890</td>
      <td>186.77418</td>
      <td>101.96023</td>
    </tr>
    <tr>
      <th>tether</th>
      <td>-0.21173</td>
      <td>0.04935</td>
      <td>0.00640</td>
      <td>-0.04237</td>
      <td>0.28037</td>
      <td>-0.00542</td>
      <td>0.01954</td>
    </tr>
    <tr>
      <th>ripple</th>
      <td>-0.37819</td>
      <td>-0.60926</td>
      <td>2.24984</td>
      <td>0.23455</td>
      <td>-17.55245</td>
      <td>39.53888</td>
      <td>-16.60193</td>
    </tr>
    <tr>
      <th>bitcoin-cash</th>
      <td>2.90585</td>
      <td>17.09717</td>
      <td>14.75334</td>
      <td>15.74903</td>
      <td>-13.71793</td>
      <td>21.66042</td>
      <td>14.49384</td>
    </tr>
    <tr>
      <th>binancecoin</th>
      <td>2.10423</td>
      <td>12.85511</td>
      <td>6.80688</td>
      <td>0.05865</td>
      <td>36.33486</td>
      <td>155.61937</td>
      <td>69.69195</td>
    </tr>
    <tr>
      <th>chainlink</th>
      <td>-0.23935</td>
      <td>20.69459</td>
      <td>9.30098</td>
      <td>-11.21747</td>
      <td>-43.69522</td>
      <td>403.22917</td>
      <td>325.13186</td>
    </tr>
    <tr>
      <th>cardano</th>
      <td>0.00322</td>
      <td>13.99302</td>
      <td>5.55476</td>
      <td>10.10553</td>
      <td>-22.84776</td>
      <td>264.51418</td>
      <td>156.09756</td>
    </tr>
    <tr>
      <th>litecoin</th>
      <td>-0.06341</td>
      <td>6.60221</td>
      <td>7.28931</td>
      <td>1.21662</td>
      <td>-17.23960</td>
      <td>27.49919</td>
      <td>-12.66408</td>
    </tr>
    <tr>
      <th>bitcoin-cash-sv</th>
      <td>0.92530</td>
      <td>3.29641</td>
      <td>-1.86656</td>
      <td>2.88926</td>
      <td>-24.87434</td>
      <td>7.42562</td>
      <td>93.73082</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Generate summary statistics
df_market_data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price_change_percentage_24h</th>
      <th>price_change_percentage_7d</th>
      <th>price_change_percentage_14d</th>
      <th>price_change_percentage_30d</th>
      <th>price_change_percentage_60d</th>
      <th>price_change_percentage_200d</th>
      <th>price_change_percentage_1y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>41.000000</td>
      <td>41.000000</td>
      <td>41.000000</td>
      <td>41.000000</td>
      <td>41.000000</td>
      <td>41.000000</td>
      <td>41.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-0.269686</td>
      <td>4.497147</td>
      <td>0.185787</td>
      <td>1.545693</td>
      <td>-0.094119</td>
      <td>236.537432</td>
      <td>347.667956</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.694793</td>
      <td>6.375218</td>
      <td>8.376939</td>
      <td>26.344218</td>
      <td>47.365803</td>
      <td>435.225304</td>
      <td>1247.842884</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-13.527860</td>
      <td>-6.094560</td>
      <td>-18.158900</td>
      <td>-34.705480</td>
      <td>-44.822480</td>
      <td>-0.392100</td>
      <td>-17.567530</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.608970</td>
      <td>0.047260</td>
      <td>-5.026620</td>
      <td>-10.438470</td>
      <td>-25.907990</td>
      <td>21.660420</td>
      <td>0.406170</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-0.063410</td>
      <td>3.296410</td>
      <td>0.109740</td>
      <td>-0.042370</td>
      <td>-7.544550</td>
      <td>83.905200</td>
      <td>69.691950</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.612090</td>
      <td>7.602780</td>
      <td>5.510740</td>
      <td>4.578130</td>
      <td>0.657260</td>
      <td>216.177610</td>
      <td>168.372510</td>
    </tr>
    <tr>
      <th>max</th>
      <td>4.840330</td>
      <td>20.694590</td>
      <td>24.239190</td>
      <td>140.795700</td>
      <td>223.064370</td>
      <td>2227.927820</td>
      <td>7852.089700</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Plot your data to see what's in your DataFrame
df_market_data.hvplot.line(
    width=800,
    height=400,
    rot=90
)
```






<div id='p3426'>
  <div id="ef7ebe5f-687b-4412-a0f1-6d4f0eea7596" data-root-id="p3426" style="display: contents;"></div>
</div>
<script type="application/javascript">(function(root) {
  var docs_json = {"07d03067-bbab-4c3a-851f-0cf0c39e4f0e":{"version":"3.2.1","title":"Bokeh Application","roots":[{"type":"object","name":"Row","id":"p3426","attributes":{"name":"Row06846","tags":["embedded"],"stylesheets":["\n:host(.pn-loading.pn-arc):before, .pn-loading.pn-arc:before {\n  background-image: url(\"data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHN0eWxlPSJtYXJnaW46IGF1dG87IGJhY2tncm91bmQ6IG5vbmU7IGRpc3BsYXk6IGJsb2NrOyBzaGFwZS1yZW5kZXJpbmc6IGF1dG87IiB2aWV3Qm94PSIwIDAgMTAwIDEwMCIgcHJlc2VydmVBc3BlY3RSYXRpbz0ieE1pZFlNaWQiPiAgPGNpcmNsZSBjeD0iNTAiIGN5PSI1MCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjYzNjM2MzIiBzdHJva2Utd2lkdGg9IjEwIiByPSIzNSIgc3Ryb2tlLWRhc2hhcnJheT0iMTY0LjkzMzYxNDMxMzQ2NDE1IDU2Ljk3Nzg3MTQzNzgyMTM4Ij4gICAgPGFuaW1hdGVUcmFuc2Zvcm0gYXR0cmlidXRlTmFtZT0idHJhbnNmb3JtIiB0eXBlPSJyb3RhdGUiIHJlcGVhdENvdW50PSJpbmRlZmluaXRlIiBkdXI9IjFzIiB2YWx1ZXM9IjAgNTAgNTA7MzYwIDUwIDUwIiBrZXlUaW1lcz0iMDsxIj48L2FuaW1hdGVUcmFuc2Zvcm0+ICA8L2NpcmNsZT48L3N2Zz4=\");\n  background-size: auto calc(min(50%, 400px));\n}",{"type":"object","name":"ImportedStyleSheet","id":"p3429","attributes":{"url":"https://cdn.holoviz.org/panel/1.2.1/dist/css/loading.css"}},{"type":"object","name":"ImportedStyleSheet","id":"p3585","attributes":{"url":"https://cdn.holoviz.org/panel/1.2.1/dist/css/listpanel.css"}},{"type":"object","name":"ImportedStyleSheet","id":"p3427","attributes":{"url":"https://cdn.holoviz.org/panel/1.2.1/dist/bundled/theme/default.css"}},{"type":"object","name":"ImportedStyleSheet","id":"p3428","attributes":{"url":"https://cdn.holoviz.org/panel/1.2.1/dist/bundled/theme/native.css"}}],"min_width":800,"margin":0,"sizing_mode":"stretch_width","align":"start","children":[{"type":"object","name":"Spacer","id":"p3430","attributes":{"name":"HSpacer06857","stylesheets":["\n:host(.pn-loading.pn-arc):before, .pn-loading.pn-arc:before {\n  background-image: url(\"data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHN0eWxlPSJtYXJnaW46IGF1dG87IGJhY2tncm91bmQ6IG5vbmU7IGRpc3BsYXk6IGJsb2NrOyBzaGFwZS1yZW5kZXJpbmc6IGF1dG87IiB2aWV3Qm94PSIwIDAgMTAwIDEwMCIgcHJlc2VydmVBc3BlY3RSYXRpbz0ieE1pZFlNaWQiPiAgPGNpcmNsZSBjeD0iNTAiIGN5PSI1MCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjYzNjM2MzIiBzdHJva2Utd2lkdGg9IjEwIiByPSIzNSIgc3Ryb2tlLWRhc2hhcnJheT0iMTY0LjkzMzYxNDMxMzQ2NDE1IDU2Ljk3Nzg3MTQzNzgyMTM4Ij4gICAgPGFuaW1hdGVUcmFuc2Zvcm0gYXR0cmlidXRlTmFtZT0idHJhbnNmb3JtIiB0eXBlPSJyb3RhdGUiIHJlcGVhdENvdW50PSJpbmRlZmluaXRlIiBkdXI9IjFzIiB2YWx1ZXM9IjAgNTAgNTA7MzYwIDUwIDUwIiBrZXlUaW1lcz0iMDsxIj48L2FuaW1hdGVUcmFuc2Zvcm0+ICA8L2NpcmNsZT48L3N2Zz4=\");\n  background-size: auto calc(min(50%, 400px));\n}",{"id":"p3429"},{"id":"p3427"},{"id":"p3428"}],"margin":0,"sizing_mode":"stretch_width","align":"start"}},{"type":"object","name":"Figure","id":"p3456","attributes":{"width":800,"height":400,"margin":[5,10],"sizing_mode":"fixed","align":"start","x_range":{"type":"object","name":"FactorRange","id":"p3431","attributes":{"tags":[[["coin_id","coin_id",null]],[]],"factors":["bitcoin","ethereum","tether","ripple","bitcoin-cash","binancecoin","chainlink","cardano","litecoin","bitcoin-cash-sv","crypto-com-chain","usd-coin","eos","monero","tron","tezos","okb","stellar","cosmos","cdai","neo","wrapped-bitcoin","leo-token","huobi-token","nem","binance-usd","iota","vechain","zcash","theta-token","dash","ethereum-classic","ethlend","maker","havven","omisego","celsius-degree-token","ontology","ftx-token","true-usd","digibyte"]}},"y_range":{"type":"object","name":"Range1d","id":"p3432","attributes":{"tags":[[["value","value",null]],{"type":"map","entries":[["invert_yaxis",false],["autorange",false]]}],"start":-834.5136980000001,"end":8641.780918,"reset_start":-834.5136980000001,"reset_end":8641.780918}},"x_scale":{"type":"object","name":"CategoricalScale","id":"p3466"},"y_scale":{"type":"object","name":"LinearScale","id":"p3467"},"title":{"type":"object","name":"Title","id":"p3459","attributes":{"text_color":"black","text_font_size":"12pt"}},"renderers":[{"type":"object","name":"GlyphRenderer","id":"p3492","attributes":{"name":"price_change_percentage_24h","data_source":{"type":"object","name":"ColumnDataSource","id":"p3483","attributes":{"selected":{"type":"object","name":"Selection","id":"p3484","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"p3485"},"data":{"type":"map","entries":[["coin_id",["bitcoin","ethereum","tether","ripple","bitcoin-cash","binancecoin","chainlink","cardano","litecoin","bitcoin-cash-sv","crypto-com-chain","usd-coin","eos","monero","tron","tezos","okb","stellar","cosmos","cdai","neo","wrapped-bitcoin","leo-token","huobi-token","nem","binance-usd","iota","vechain","zcash","theta-token","dash","ethereum-classic","ethlend","maker","havven","omisego","celsius-degree-token","ontology","ftx-token","true-usd","digibyte"]],["value",{"type":"ndarray","array":{"type":"bytes","data":"nZ0MjpJX8T8MzXUaaanMP5mByvj3Gcu/wCZr1EM02L8ep+hILj8HQL0Yyol21QBAwhcmUwWjzr//CS5W1GBqP7JGPUSjO7C/QKTfvg6c7T8urYbEPZbjPzeJQWDl0Ma/1pC4x9KHwj87NgLxun7bP90HILWJk7M/JjYf14aK5b83iUFg5dAFwG3i5H6HIvC/5nlwd9Zu7r9i83FtqBjLP+iHEcKjjd8/M9yAzw+j8T/fGtgqweLAv+RmuAGfH9q/UdobfGEy678D7KNTVz67v8YzaOif4No/teBFX0Ga9D8G2Eenrnzjvx+duvJZPhLAoMN8eQH28L+cxCCwcmjdv667eapDDivAX5hMFYxK479EUaBP5EkQwAZkr3d/XBNA0JuKVBgbBEDqBDQRNrz1vxo09E9wseo/tI6qJoi6r79qMA3DR8QHQA=="},"shape":[41],"dtype":"float64","order":"little"}],["Variable",["price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h"]]]}}},"view":{"type":"object","name":"CDSView","id":"p3493","attributes":{"filter":{"type":"object","name":"AllIndices","id":"p3494"}}},"glyph":{"type":"object","name":"Line","id":"p3489","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"coin_id"},"y":{"type":"field","field":"value"},"line_color":"#30a2da","line_width":2}},"selection_glyph":{"type":"object","name":"Line","id":"p3497","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"coin_id"},"y":{"type":"field","field":"value"},"line_color":"#30a2da","line_width":2}},"nonselection_glyph":{"type":"object","name":"Line","id":"p3490","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"coin_id"},"y":{"type":"field","field":"value"},"line_color":"#30a2da","line_alpha":0.1,"line_width":2}},"muted_glyph":{"type":"object","name":"Line","id":"p3491","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"coin_id"},"y":{"type":"field","field":"value"},"line_color":"#30a2da","line_alpha":0.2,"line_width":2}}}},{"type":"object","name":"GlyphRenderer","id":"p3507","attributes":{"name":"price_change_percentage_7d","data_source":{"type":"object","name":"ColumnDataSource","id":"p3498","attributes":{"selected":{"type":"object","name":"Selection","id":"p3499","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"p3500"},"data":{"type":"map","entries":[["coin_id",["bitcoin","ethereum","tether","ripple","bitcoin-cash","binancecoin","chainlink","cardano","litecoin","bitcoin-cash-sv","crypto-com-chain","usd-coin","eos","monero","tron","tezos","okb","stellar","cosmos","cdai","neo","wrapped-bitcoin","leo-token","huobi-token","nem","binance-usd","iota","vechain","zcash","theta-token","dash","ethereum-classic","ethlend","maker","havven","omisego","celsius-degree-token","ontology","ftx-token","true-usd","digibyte"]],["value",{"type":"ndarray","array":{"type":"bytes","data":"RbsKKT9pHkAzUBn/PsMkQL99HThnRKk/yVnY0w5/47/+JhQi4BgxQBL3WPrQtSlA7yB2ptCxNEA/jBAebfwrQPFL/bypaBpArKjBNAxfCkByv0NRoK8WwPlJtU/HY76/lPsdigL99L+n6Egu/5EvQE1KQbeX9BBA0m9fB87pIUBlU67wLjcSwKbtX1lpkgBAZwqd19gVMEBcIEHxY8ytP3wnZr0YigNAOh4zUBmfHUD3Hi457pT1vzNQGf8+4/k/kQ96Nqs+6781Y9F0djKoP2dEaW/wRRhAgXhdv2A3/L/8GHPXErIkQP8JLlbUYBjAVU0QdR9gFEDRlnMprmoIQGCrBIvD2RBAAiuHFtlOIECbG9MTllgRQKhXyjLEURtAqn06HjNQ4z+UvDrHgGzzvwltOZfiahxAsD2zJEBNxT8yj/zBwHPlPw=="},"shape":[41],"dtype":"float64","order":"little"}],["Variable",["price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d"]]]}}},"view":{"type":"object","name":"CDSView","id":"p3508","attributes":{"filter":{"type":"object","name":"AllIndices","id":"p3509"}}},"glyph":{"type":"object","name":"Line","id":"p3504","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"coin_id"},"y":{"type":"field","field":"value"},"line_color":"#fc4f30","line_width":2}},"selection_glyph":{"type":"object","name":"Line","id":"p3511","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"coin_id"},"y":{"type":"field","field":"value"},"line_color":"#fc4f30","line_width":2}},"nonselection_glyph":{"type":"object","name":"Line","id":"p3505","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"coin_id"},"y":{"type":"field","field":"value"},"line_color":"#fc4f30","line_alpha":0.1,"line_width":2}},"muted_glyph":{"type":"object","name":"Line","id":"p3506","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"coin_id"},"y":{"type":"field","field":"value"},"line_color":"#fc4f30","line_alpha":0.2,"line_width":2}}}},{"type":"object","name":"GlyphRenderer","id":"p3521","attributes":{"name":"price_change_percentage_14d","data_source":{"type":"object","name":"ColumnDataSource","id":"p3512","attributes":{"selected":{"type":"object","name":"Selection","id":"p3513","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"p3514"},"data":{"type":"map","entries":[["coin_id",["bitcoin","ethereum","tether","ripple","bitcoin-cash","binancecoin","chainlink","cardano","litecoin","bitcoin-cash-sv","crypto-com-chain","usd-coin","eos","monero","tron","tezos","okb","stellar","cosmos","cdai","neo","wrapped-bitcoin","leo-token","huobi-token","nem","binance-usd","iota","vechain","zcash","theta-token","dash","ethereum-classic","ethlend","maker","havven","omisego","celsius-degree-token","ontology","ftx-token","true-usd","digibyte"]],["value",{"type":"ndarray","array":{"type":"bytes","data":"zQaZZORMGkB5knTN5DsTQC1DHOviNno/ctwpHaz/AUCLic3HtYEtQJgvL8A+OhtACoDxDBqaIkAOhGQBEzgWQICfceFAKB1AIVnABG7d/b8xsfm4NhQhwGtI3GPpQ3e/pfeNrz0z8j8ao3VUNWkyQBUA4xk0dPY/VIzzN6GQAkCXrfVFQhsUwKpla32RUPG/Qgkzbf8KFkDPg7uzdtvFP8MN+PwwsiPAYFlpUgo6GkB3+GuyRh0cwM2v5gDBHN0/jliLTwEwBUDWrZ6T3jeuP7pOIy2VNw7Am1Wfq60oMsAJM23/yioXQJfK2xFOSxrAQE0tW+uL4D9OucK7XET0v0SLbOf7mSPA/pqsUQ/R978vaYzWUZUqwDtT6LzGbirAvodLjjs9OEByUMJM2x8lwJ30vvG1Z8q/ZCMQr+sXvD+7D0BqE0cdwA=="},"shape":[41],"dtype":"float64","order":"little"}],["Variable",["price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d"]]]}}},"view":{"type":"object","name":"CDSView","id":"p3522","attributes":{"filter":{"type":"object","name":"AllIndices","id":"p3523"}}},"glyph":{"type":"object","name":"Line","id":"p3518","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"coin_id"},"y":{"type":"field","field":"value"},"line_color":"#e5ae38","line_width":2}},"selection_glyph":{"type":"object","name":"Line","id":"p3525","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"coin_id"},"y":{"type":"field","field":"value"},"line_color":"#e5ae38","line_width":2}},"nonselection_glyph":{"type":"object","name":"Line","id":"p3519","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"coin_id"},"y":{"type":"field","field":"value"},"line_color":"#e5ae38","line_alpha":0.1,"line_width":2}},"muted_glyph":{"type":"object","name":"Line","id":"p3520","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"coin_id"},"y":{"type":"field","field":"value"},"line_color":"#e5ae38","line_alpha":0.2,"line_width":2}}}},{"type":"object","name":"GlyphRenderer","id":"p3535","attributes":{"name":"price_change_percentage_30d","data_source":{"type":"object","name":"ColumnDataSource","id":"p3526","attributes":{"selected":{"type":"object","name":"Selection","id":"p3527","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"p3528"},"data":{"type":"map","entries":[["coin_id",["bitcoin","ethereum","tether","ripple","bitcoin-cash","binancecoin","chainlink","cardano","litecoin","bitcoin-cash-sv","crypto-com-chain","usd-coin","eos","monero","tron","tezos","okb","stellar","cosmos","cdai","neo","wrapped-bitcoin","leo-token","huobi-token","nem","binance-usd","iota","vechain","zcash","theta-token","dash","ethereum-classic","ethlend","maker","havven","omisego","celsius-degree-token","ontology","ftx-token","true-usd","digibyte"]],["value",{"type":"ndarray","array":{"type":"bytes","data":"VMa/z7iwHkAl6ZrJN9vAP/28qUiFsaW/L26jAbwFzj8dcjPcgH8vQPMf0m9fB64/5bhTOlhvJsBnfjUHCDYkQGWNeohGd/M/sCDNWDQdB0Cmft5UpHIxwJgvL8A+OsW/t0WZDTKJEsDeVKTC2HpDQEzD8BExNSnAjWK5pdVALMDjjcwjf+AkwIrIsIo3QiDAjliLTwFQEkAWpBmLpnMBwE3WqIdo9DXAVG8NbJWAHUAnMQisHJoIQGA8g4b+CQrACcTr+gU7DcCPpQ9dUN+SP80Bgjl6nBDA9S1zuixWLMBC7Eyh83odQEku/yH9bj9AtTf4wmQqH8Am/FI/byoHQNjYJaq3/j3A9GxWfa62MECdRloqb9c0wF4R/G8lOyLArK3YX3aZYUC1/SsrTVpBwP/PYb68ICXAFR3J5T+knz+ndLD+z4EqQA=="},"shape":[41],"dtype":"float64","order":"little"}],["Variable",["price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d"]]]}}},"view":{"type":"object","name":"CDSView","id":"p3536","attributes":{"filter":{"type":"object","name":"AllIndices","id":"p3537"}}},"glyph":{"type":"object","name":"Line","id":"p3532","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"coin_id"},"y":{"type":"field","field":"value"},"line_color":"#6d904f","line_width":2}},"selection_glyph":{"type":"object","name":"Line","id":"p3539","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"coin_id"},"y":{"type":"field","field":"value"},"line_color":"#6d904f","line_width":2}},"nonselection_glyph":{"type":"object","name":"Line","id":"p3533","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"coin_id"},"y":{"type":"field","field":"value"},"line_color":"#6d904f","line_alpha":0.1,"line_width":2}},"muted_glyph":{"type":"object","name":"Line","id":"p3534","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"coin_id"},"y":{"type":"field","field":"value"},"line_color":"#6d904f","line_alpha":0.2,"line_width":2}}}},{"type":"object","name":"GlyphRenderer","id":"p3549","attributes":{"name":"price_change_percentage_60d","data_source":{"type":"object","name":"ColumnDataSource","id":"p3540","attributes":{"selected":{"type":"object","name":"Selection","id":"p3541","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"p3542"},"data":{"type":"map","entries":[["coin_id",["bitcoin","ethereum","tether","ripple","bitcoin-cash","binancecoin","chainlink","cardano","litecoin","bitcoin-cash-sv","crypto-com-chain","usd-coin","eos","monero","tron","tezos","okb","stellar","cosmos","cdai","neo","wrapped-bitcoin","leo-token","huobi-token","nem","binance-usd","iota","vechain","zcash","theta-token","dash","ethereum-classic","ethlend","maker","havven","omisego","celsius-degree-token","ontology","ftx-token","true-usd","digibyte"]],["value",{"type":"ndarray","array":{"type":"bytes","data":"FvvL7skDCsAs1JrmHccpwKjjMQOV8dE/2qz6XG2NMcBzol2FlG8rwIleRrHcKkJAN8MN+PzYRcBi26LMBtk2wCEf9GxWPTHAq5UJv9TfOMBg5dAi23kwwDygbMoV3qU/l631RUIrPsDNzMzMzNxEQJm7lpAPGhZAms5OBkdpRsBKe4MvTKYGwEsfuqC+zT7APL1SliHOHMC4AZ8fRgjlP9L7xteeCStAFqQZi6azDMCvJeSDni0ewDeOWItPQQvAXvQVpBm3VEDaOGItPgW4PwpLPKBsQkHAaw4QzNHPRcAJM23/yoo0wBmQvd79AVRAfa62Yn85OsBVGFsIcug5wLCsNCkFHQFA529CIQKeMsA0uoPYmbZDwPKwUGua0VNA/pqsUQ/ia0Ao8iTpmllAwFuxv+yenBdAmrFoOjsZ0D8s1JrmHRc/wA=="},"shape":[41],"dtype":"float64","order":"little"}],["Variable",["price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d"]]]}}},"view":{"type":"object","name":"CDSView","id":"p3550","attributes":{"filter":{"type":"object","name":"AllIndices","id":"p3551"}}},"glyph":{"type":"object","name":"Line","id":"p3546","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"coin_id"},"y":{"type":"field","field":"value"},"line_color":"#8b8b8b","line_width":2}},"selection_glyph":{"type":"object","name":"Line","id":"p3553","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"coin_id"},"y":{"type":"field","field":"value"},"line_color":"#8b8b8b","line_width":2}},"nonselection_glyph":{"type":"object","name":"Line","id":"p3547","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"coin_id"},"y":{"type":"field","field":"value"},"line_color":"#8b8b8b","line_alpha":0.1,"line_width":2}},"muted_glyph":{"type":"object","name":"Line","id":"p3548","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"coin_id"},"y":{"type":"field","field":"value"},"line_color":"#8b8b8b","line_alpha":0.2,"line_width":2}}}},{"type":"object","name":"GlyphRenderer","id":"p3563","attributes":{"name":"price_change_percentage_200d","data_source":{"type":"object","name":"ColumnDataSource","id":"p3554","attributes":{"selected":{"type":"object","name":"Selection","id":"p3555","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"p3556"},"data":{"type":"map","entries":[["coin_id",["bitcoin","ethereum","tether","ripple","bitcoin-cash","binancecoin","chainlink","cardano","litecoin","bitcoin-cash-sv","crypto-com-chain","usd-coin","eos","monero","tron","tezos","okb","stellar","cosmos","cdai","neo","wrapped-bitcoin","leo-token","huobi-token","nem","binance-usd","iota","vechain","zcash","theta-token","dash","ethereum-classic","ethlend","maker","havven","omisego","celsius-degree-token","ontology","ftx-token","true-usd","digibyte"]],["value",{"type":"ndarray","array":{"type":"bytes","data":"xY8xdy3hVED0piIVxlhnQBe86CtIM3a/9zsUBfrEQ0DTn/1IEak1QPTDCOHRc2NAl3MprqozeUAexM4UOohwQIB9dOrKfztAFhiyutWzHUCN7iB2plZsQIQqNXugFcS/ZHWr56QPMkALe9rhrzBlQH+8V61MnGBA5nlwd9a2RUARHm0csfpDQC+Lic3HJ1VAkj8YeO4/Z0AOvjCZKhjZv2vUQzS61GNAtRX7y+75VEC5GW7A52cwQAPso1NXHkVAKa4q+64Fa0A7NgLxun65P+TaUDHOwVdAyM1wAz7bcEAFwHgGDaNeQH3Qs1k1lYtAOUVHcvnvGEANGvonuMgkQCYZOQvbZ6FAGD4ipkSiVED5MeauZXeDQHi0ccQaSoJAC170FcTYmEB1PGagMmJLQC2yne+nvkxAndfYJaq3tr9R9wFIbSl+QA=="},"shape":[41],"dtype":"float64","order":"little"}],["Variable",["price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d"]]]}}},"view":{"type":"object","name":"CDSView","id":"p3564","attributes":{"filter":{"type":"object","name":"AllIndices","id":"p3565"}}},"glyph":{"type":"object","name":"Line","id":"p3560","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"coin_id"},"y":{"type":"field","field":"value"},"line_color":"#17becf","line_width":2}},"selection_glyph":{"type":"object","name":"Line","id":"p3567","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"coin_id"},"y":{"type":"field","field":"value"},"line_color":"#17becf","line_width":2}},"nonselection_glyph":{"type":"object","name":"Line","id":"p3561","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"coin_id"},"y":{"type":"field","field":"value"},"line_color":"#17becf","line_alpha":0.1,"line_width":2}},"muted_glyph":{"type":"object","name":"Line","id":"p3562","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"coin_id"},"y":{"type":"field","field":"value"},"line_color":"#17becf","line_alpha":0.2,"line_width":2}}}},{"type":"object","name":"GlyphRenderer","id":"p3577","attributes":{"name":"price_change_percentage_1y","data_source":{"type":"object","name":"ColumnDataSource","id":"p3568","attributes":{"selected":{"type":"object","name":"Selection","id":"p3569","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"p3570"},"data":{"type":"map","entries":[["coin_id",["bitcoin","ethereum","tether","ripple","bitcoin-cash","binancecoin","chainlink","cardano","litecoin","bitcoin-cash-sv","crypto-com-chain","usd-coin","eos","monero","tron","tezos","okb","stellar","cosmos","cdai","neo","wrapped-bitcoin","leo-token","huobi-token","nem","binance-usd","iota","vechain","zcash","theta-token","dash","ethereum-classic","ethlend","maker","havven","omisego","celsius-degree-token","ontology","ftx-token","true-usd","digibyte"]],["value",{"type":"ndarray","array":{"type":"bytes","data":"kQpjC0HCQkDWqIdodH1ZQN/42jNLApQ/NXugFRiaMMDN6bKY2PwsQOcdp+hIbFFAaTo7GRxSdEC6LCY2H4NjQPnaM0sCVCnA0NA/wcVuV0CQvd798RBzQL4wmSoYlci/t39lpUmRMcDovMYuUaFhQEPKT6p9nk1A6PaSxmiAYUBdUN8yp75hQMzuycNCnStAw7ZFmQ2cVEAHsTOFzmvSvx/0bFZ9aWBAhhvw+WHEQkAN/RNcrAA1QN8Vwf9WkjhAS7A4nPkWaUAVUn5S7dPBPzUk7rH04Q/AIsMq3shbaUBf0hito7hUQPyMCwcC64VAzsKedvirA8Dwoq8gzTgmQHRGlPYWrL5Ayk+qfTpOJUDfiVkvRjaFQEfJq3MMC3RAnFCIgONmn0CsVib8Uk8pwNttF5rrC2VAyXGndLD+2T/ZfFwbKgxnQA=="},"shape":[41],"dtype":"float64","order":"little"}],["Variable",["price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y"]]]}}},"view":{"type":"object","name":"CDSView","id":"p3578","attributes":{"filter":{"type":"object","name":"AllIndices","id":"p3579"}}},"glyph":{"type":"object","name":"Line","id":"p3574","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"coin_id"},"y":{"type":"field","field":"value"},"line_color":"#9467bd","line_width":2}},"selection_glyph":{"type":"object","name":"Line","id":"p3581","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"coin_id"},"y":{"type":"field","field":"value"},"line_color":"#9467bd","line_width":2}},"nonselection_glyph":{"type":"object","name":"Line","id":"p3575","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"coin_id"},"y":{"type":"field","field":"value"},"line_color":"#9467bd","line_alpha":0.1,"line_width":2}},"muted_glyph":{"type":"object","name":"Line","id":"p3576","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"coin_id"},"y":{"type":"field","field":"value"},"line_color":"#9467bd","line_alpha":0.2,"line_width":2}}}}],"toolbar":{"type":"object","name":"Toolbar","id":"p3465","attributes":{"tools":[{"type":"object","name":"WheelZoomTool","id":"p3436","attributes":{"tags":["hv_created"],"zoom_together":"none"}},{"type":"object","name":"HoverTool","id":"p3437","attributes":{"tags":["hv_created"],"renderers":[{"id":"p3492"},{"id":"p3507"},{"id":"p3521"},{"id":"p3535"},{"id":"p3549"},{"id":"p3563"},{"id":"p3577"}],"tooltips":[["Variable","@{Variable}"],["coin_id","@{coin_id}"],["value","@{value}"]]}},{"type":"object","name":"SaveTool","id":"p3478"},{"type":"object","name":"PanTool","id":"p3479"},{"type":"object","name":"BoxZoomTool","id":"p3480","attributes":{"overlay":{"type":"object","name":"BoxAnnotation","id":"p3481","attributes":{"syncable":false,"level":"overlay","visible":false,"left_units":"canvas","right_units":"canvas","bottom_units":"canvas","top_units":"canvas","line_color":"black","line_alpha":1.0,"line_width":2,"line_dash":[4,4],"fill_color":"lightgrey","fill_alpha":0.5}}}},{"type":"object","name":"ResetTool","id":"p3482"}],"active_drag":{"id":"p3479"},"active_scroll":{"id":"p3436"}}},"left":[{"type":"object","name":"LinearAxis","id":"p3473","attributes":{"ticker":{"type":"object","name":"BasicTicker","id":"p3474","attributes":{"mantissas":[1,2,5]}},"formatter":{"type":"object","name":"BasicTickFormatter","id":"p3475"},"axis_label":"","major_label_policy":{"type":"object","name":"AllLabels","id":"p3476"}}}],"right":[{"type":"object","name":"Legend","id":"p3495","attributes":{"location":[0,0],"title":"Variable","click_policy":"mute","items":[{"type":"object","name":"LegendItem","id":"p3496","attributes":{"label":{"type":"value","value":"price_change_percentage_24h"},"renderers":[{"id":"p3492"}]}},{"type":"object","name":"LegendItem","id":"p3510","attributes":{"label":{"type":"value","value":"price_change_percentage_7d"},"renderers":[{"id":"p3507"}]}},{"type":"object","name":"LegendItem","id":"p3524","attributes":{"label":{"type":"value","value":"price_change_percentage_14d"},"renderers":[{"id":"p3521"}]}},{"type":"object","name":"LegendItem","id":"p3538","attributes":{"label":{"type":"value","value":"price_change_percentage_30d"},"renderers":[{"id":"p3535"}]}},{"type":"object","name":"LegendItem","id":"p3552","attributes":{"label":{"type":"value","value":"price_change_percentage_60d"},"renderers":[{"id":"p3549"}]}},{"type":"object","name":"LegendItem","id":"p3566","attributes":{"label":{"type":"value","value":"price_change_percentage_200d"},"renderers":[{"id":"p3563"}]}},{"type":"object","name":"LegendItem","id":"p3580","attributes":{"label":{"type":"value","value":"price_change_percentage_1y"},"renderers":[{"id":"p3577"}]}}]}}],"below":[{"type":"object","name":"CategoricalAxis","id":"p3468","attributes":{"ticker":{"type":"object","name":"CategoricalTicker","id":"p3469"},"formatter":{"type":"object","name":"CategoricalTickFormatter","id":"p3470"},"axis_label":"coin_id","major_label_orientation":1.5707963267948966,"major_label_policy":{"type":"object","name":"AllLabels","id":"p3471"}}}],"center":[{"type":"object","name":"Grid","id":"p3472","attributes":{"axis":{"id":"p3468"},"grid_line_color":null}},{"type":"object","name":"Grid","id":"p3477","attributes":{"dimension":1,"axis":{"id":"p3473"},"grid_line_color":null}}],"min_border_top":10,"min_border_bottom":10,"min_border_left":10,"min_border_right":10,"output_backend":"webgl"}},{"type":"object","name":"Spacer","id":"p3583","attributes":{"name":"HSpacer06860","stylesheets":["\n:host(.pn-loading.pn-arc):before, .pn-loading.pn-arc:before {\n  background-image: url(\"data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHN0eWxlPSJtYXJnaW46IGF1dG87IGJhY2tncm91bmQ6IG5vbmU7IGRpc3BsYXk6IGJsb2NrOyBzaGFwZS1yZW5kZXJpbmc6IGF1dG87IiB2aWV3Qm94PSIwIDAgMTAwIDEwMCIgcHJlc2VydmVBc3BlY3RSYXRpbz0ieE1pZFlNaWQiPiAgPGNpcmNsZSBjeD0iNTAiIGN5PSI1MCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjYzNjM2MzIiBzdHJva2Utd2lkdGg9IjEwIiByPSIzNSIgc3Ryb2tlLWRhc2hhcnJheT0iMTY0LjkzMzYxNDMxMzQ2NDE1IDU2Ljk3Nzg3MTQzNzgyMTM4Ij4gICAgPGFuaW1hdGVUcmFuc2Zvcm0gYXR0cmlidXRlTmFtZT0idHJhbnNmb3JtIiB0eXBlPSJyb3RhdGUiIHJlcGVhdENvdW50PSJpbmRlZmluaXRlIiBkdXI9IjFzIiB2YWx1ZXM9IjAgNTAgNTA7MzYwIDUwIDUwIiBrZXlUaW1lcz0iMDsxIj48L2FuaW1hdGVUcmFuc2Zvcm0+ICA8L2NpcmNsZT48L3N2Zz4=\");\n  background-size: auto calc(min(50%, 400px));\n}",{"id":"p3429"},{"id":"p3427"},{"id":"p3428"}],"margin":0,"sizing_mode":"stretch_width","align":"start"}}]}}],"defs":[{"type":"model","name":"ReactiveHTML1"},{"type":"model","name":"FlexBox1","properties":[{"name":"align_content","kind":"Any","default":"flex-start"},{"name":"align_items","kind":"Any","default":"flex-start"},{"name":"flex_direction","kind":"Any","default":"row"},{"name":"flex_wrap","kind":"Any","default":"wrap"},{"name":"justify_content","kind":"Any","default":"flex-start"}]},{"type":"model","name":"FloatPanel1","properties":[{"name":"config","kind":"Any","default":{"type":"map"}},{"name":"contained","kind":"Any","default":true},{"name":"position","kind":"Any","default":"right-top"},{"name":"offsetx","kind":"Any","default":null},{"name":"offsety","kind":"Any","default":null},{"name":"theme","kind":"Any","default":"primary"},{"name":"status","kind":"Any","default":"normalized"}]},{"type":"model","name":"GridStack1","properties":[{"name":"mode","kind":"Any","default":"warn"},{"name":"ncols","kind":"Any","default":null},{"name":"nrows","kind":"Any","default":null},{"name":"allow_resize","kind":"Any","default":true},{"name":"allow_drag","kind":"Any","default":true},{"name":"state","kind":"Any","default":[]}]},{"type":"model","name":"drag1","properties":[{"name":"slider_width","kind":"Any","default":5},{"name":"slider_color","kind":"Any","default":"black"},{"name":"value","kind":"Any","default":50}]},{"type":"model","name":"click1","properties":[{"name":"terminal_output","kind":"Any","default":""},{"name":"debug_name","kind":"Any","default":""},{"name":"clears","kind":"Any","default":0}]},{"type":"model","name":"FastWrapper1","properties":[{"name":"object","kind":"Any","default":null},{"name":"style","kind":"Any","default":null}]},{"type":"model","name":"NotificationAreaBase1","properties":[{"name":"js_events","kind":"Any","default":{"type":"map"}},{"name":"position","kind":"Any","default":"bottom-right"},{"name":"_clear","kind":"Any","default":0}]},{"type":"model","name":"NotificationArea1","properties":[{"name":"js_events","kind":"Any","default":{"type":"map"}},{"name":"notifications","kind":"Any","default":[]},{"name":"position","kind":"Any","default":"bottom-right"},{"name":"_clear","kind":"Any","default":0},{"name":"types","kind":"Any","default":[{"type":"map","entries":[["type","warning"],["background","#ffc107"],["icon",{"type":"map","entries":[["className","fas fa-exclamation-triangle"],["tagName","i"],["color","white"]]}]]},{"type":"map","entries":[["type","info"],["background","#007bff"],["icon",{"type":"map","entries":[["className","fas fa-info-circle"],["tagName","i"],["color","white"]]}]]}]}]},{"type":"model","name":"Notification","properties":[{"name":"background","kind":"Any","default":null},{"name":"duration","kind":"Any","default":3000},{"name":"icon","kind":"Any","default":null},{"name":"message","kind":"Any","default":""},{"name":"notification_type","kind":"Any","default":null},{"name":"_destroyed","kind":"Any","default":false}]},{"type":"model","name":"TemplateActions1","properties":[{"name":"open_modal","kind":"Any","default":0},{"name":"close_modal","kind":"Any","default":0}]},{"type":"model","name":"BootstrapTemplateActions1","properties":[{"name":"open_modal","kind":"Any","default":0},{"name":"close_modal","kind":"Any","default":0}]},{"type":"model","name":"MaterialTemplateActions1","properties":[{"name":"open_modal","kind":"Any","default":0},{"name":"close_modal","kind":"Any","default":0}]}]}};
  var render_items = [{"docid":"07d03067-bbab-4c3a-851f-0cf0c39e4f0e","roots":{"p3426":"ef7ebe5f-687b-4412-a0f1-6d4f0eea7596"},"root_ids":["p3426"]}];
  var docs = Object.values(docs_json)
  if (!docs) {
    return
  }
  const py_version = docs[0].version.replace('rc', '-rc.').replace('.dev', '-dev.')
  const is_dev = py_version.indexOf("+") !== -1 || py_version.indexOf("-") !== -1
  function embed_document(root) {
    var Bokeh = get_bokeh(root)
    Bokeh.embed.embed_items_notebook(docs_json, render_items);
    for (const render_item of render_items) {
      for (const root_id of render_item.root_ids) {
	const id_el = document.getElementById(root_id)
	if (id_el.children.length && (id_el.children[0].className === 'bk-root')) {
	  const root_el = id_el.children[0]
	  root_el.id = root_el.id + '-rendered'
	}
      }
    }
  }
  function get_bokeh(root) {
    if (root.Bokeh === undefined) {
      return null
    } else if (root.Bokeh.version !== py_version && !is_dev) {
      if (root.Bokeh.versions === undefined || !root.Bokeh.versions.has(py_version)) {
	return null
      }
      return root.Bokeh.versions.get(py_version);
    } else if (root.Bokeh.version === py_version) {
      return root.Bokeh
    }
    return null
  }
  function is_loaded(root) {
    var Bokeh = get_bokeh(root)
    return (Bokeh != null && Bokeh.Panel !== undefined)
  }
  if (is_loaded(root)) {
    embed_document(root);
  } else {
    var attempts = 0;
    var timer = setInterval(function(root) {
      if (is_loaded(root)) {
        clearInterval(timer);
        embed_document(root);
      } else if (document.readyState == "complete") {
        attempts++;
        if (attempts > 200) {
          clearInterval(timer);
	  var Bokeh = get_bokeh(root)
	  if (Bokeh == null || Bokeh.Panel == null) {
            console.warn("Panel: ERROR: Unable to run Panel code because Bokeh or Panel library is missing");
	  } else {
	    console.warn("Panel: WARNING: Attempting to render but not all required libraries could be resolved.")
	    embed_document(root)
	  }
        }
      }
    }, 25, root)
  }
})(window);</script>



---

### Prepare the Data

This section prepares the data before running the K-Means algorithm. It follows these steps:

1. Use the `StandardScaler` module from scikit-learn to normalize the CSV file data. This will require you to utilize the `fit_transform` function.

2. Create a DataFrame that contains the scaled data. Be sure to set the `coin_id` index from the original DataFrame as the index for the new DataFrame. Review the resulting DataFrame.



```python
# Use the `StandardScaler()` module from scikit-learn to normalize the data from the CSV file
scaled_data = StandardScaler().fit_transform(df_market_data)
```


```python
# Create a DataFrame with the scaled data
df_market_data_scaled = pd.DataFrame(
    scaled_data,
    columns=df_market_data.columns
)

# Copy the crypto names from the original data
df_market_data_scaled["coin_id"] = df_market_data.index

# Set the coinid column as index
df_market_data_scaled = df_market_data_scaled.set_index("coin_id")

# Display sample data
df_market_data_scaled.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price_change_percentage_24h</th>
      <th>price_change_percentage_7d</th>
      <th>price_change_percentage_14d</th>
      <th>price_change_percentage_30d</th>
      <th>price_change_percentage_60d</th>
      <th>price_change_percentage_200d</th>
      <th>price_change_percentage_1y</th>
    </tr>
    <tr>
      <th>coin_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>bitcoin</th>
      <td>0.508529</td>
      <td>0.493193</td>
      <td>0.772200</td>
      <td>0.235460</td>
      <td>-0.067495</td>
      <td>-0.355953</td>
      <td>-0.251637</td>
    </tr>
    <tr>
      <th>ethereum</th>
      <td>0.185446</td>
      <td>0.934445</td>
      <td>0.558692</td>
      <td>-0.054341</td>
      <td>-0.273483</td>
      <td>-0.115759</td>
      <td>-0.199352</td>
    </tr>
    <tr>
      <th>tether</th>
      <td>0.021774</td>
      <td>-0.706337</td>
      <td>-0.021680</td>
      <td>-0.061030</td>
      <td>0.008005</td>
      <td>-0.550247</td>
      <td>-0.282061</td>
    </tr>
    <tr>
      <th>ripple</th>
      <td>-0.040764</td>
      <td>-0.810928</td>
      <td>0.249458</td>
      <td>-0.050388</td>
      <td>-0.373164</td>
      <td>-0.458259</td>
      <td>-0.295546</td>
    </tr>
    <tr>
      <th>bitcoin-cash</th>
      <td>1.193036</td>
      <td>2.000959</td>
      <td>1.760610</td>
      <td>0.545842</td>
      <td>-0.291203</td>
      <td>-0.499848</td>
      <td>-0.270317</td>
    </tr>
  </tbody>
</table>
</div>



---

### Find the Best Value for k Using the Original Data

In this section, you will use the elbow method to find the best value for `k`.

1. Code the elbow method algorithm to find the best value for `k`. Use a range from 1 to 11. 

2. Plot a line chart with all the inertia values computed with the different values of `k` to visually identify the optimal value for `k`.

3. Answer the following question: What is the best value for `k`?


```python
# Create a list with the number of k-values to try
# Use a range from 1 to 11
k = list(range(1, 11))
```


```python
# Create an empy list to store the inertia values
inertia =[]
```


```python
# Create a for loop to compute the inertia with each possible value of k
# Inside the loop:
# 1. Create a KMeans model using the loop counter for the n_clusters
# 2. Fit the model to the data using `df_market_data_scaled`
# 3. Append the model.inertia_ to the inertia list
for i in k:
    model = KMeans(n_clusters=i, random_state=0)
    model.fit(df_market_data_scaled)
    inertia.append(model.inertia_)
                   
```


```python
df_elbow = pd.DataFrame(elbow_data)
```


```python
# Create a dictionary with the data to plot the Elbow curve
elbow_data = {
    "k" : k, 
    "inertia": inertia
}


```


```python
# Plot a line chart with all the inertia values computed with 
# the different values of k to visually identify the optimal value for k.
df_elbow.hvplot.line(x="k", y="inertia", title="Elbow Curve", xticks=k)
```






<div id='p3594'>
  <div id="fed44dbe-8ef3-40cc-ae73-b13d0b69db86" data-root-id="p3594" style="display: contents;"></div>
</div>
<script type="application/javascript">(function(root) {
  var docs_json = {"d19d7030-1171-4339-8dc1-d613a1d576a4":{"version":"3.2.1","title":"Bokeh Application","roots":[{"type":"object","name":"Row","id":"p3594","attributes":{"name":"Row07070","tags":["embedded"],"stylesheets":["\n:host(.pn-loading.pn-arc):before, .pn-loading.pn-arc:before {\n  background-image: url(\"data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHN0eWxlPSJtYXJnaW46IGF1dG87IGJhY2tncm91bmQ6IG5vbmU7IGRpc3BsYXk6IGJsb2NrOyBzaGFwZS1yZW5kZXJpbmc6IGF1dG87IiB2aWV3Qm94PSIwIDAgMTAwIDEwMCIgcHJlc2VydmVBc3BlY3RSYXRpbz0ieE1pZFlNaWQiPiAgPGNpcmNsZSBjeD0iNTAiIGN5PSI1MCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjYzNjM2MzIiBzdHJva2Utd2lkdGg9IjEwIiByPSIzNSIgc3Ryb2tlLWRhc2hhcnJheT0iMTY0LjkzMzYxNDMxMzQ2NDE1IDU2Ljk3Nzg3MTQzNzgyMTM4Ij4gICAgPGFuaW1hdGVUcmFuc2Zvcm0gYXR0cmlidXRlTmFtZT0idHJhbnNmb3JtIiB0eXBlPSJyb3RhdGUiIHJlcGVhdENvdW50PSJpbmRlZmluaXRlIiBkdXI9IjFzIiB2YWx1ZXM9IjAgNTAgNTA7MzYwIDUwIDUwIiBrZXlUaW1lcz0iMDsxIj48L2FuaW1hdGVUcmFuc2Zvcm0+ICA8L2NpcmNsZT48L3N2Zz4=\");\n  background-size: auto calc(min(50%, 400px));\n}",{"type":"object","name":"ImportedStyleSheet","id":"p3597","attributes":{"url":"https://cdn.holoviz.org/panel/1.2.1/dist/css/loading.css"}},{"type":"object","name":"ImportedStyleSheet","id":"p3650","attributes":{"url":"https://cdn.holoviz.org/panel/1.2.1/dist/css/listpanel.css"}},{"type":"object","name":"ImportedStyleSheet","id":"p3595","attributes":{"url":"https://cdn.holoviz.org/panel/1.2.1/dist/bundled/theme/default.css"}},{"type":"object","name":"ImportedStyleSheet","id":"p3596","attributes":{"url":"https://cdn.holoviz.org/panel/1.2.1/dist/bundled/theme/native.css"}}],"min_width":700,"margin":0,"sizing_mode":"stretch_width","align":"start","children":[{"type":"object","name":"Spacer","id":"p3598","attributes":{"name":"HSpacer07081","stylesheets":["\n:host(.pn-loading.pn-arc):before, .pn-loading.pn-arc:before {\n  background-image: url(\"data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHN0eWxlPSJtYXJnaW46IGF1dG87IGJhY2tncm91bmQ6IG5vbmU7IGRpc3BsYXk6IGJsb2NrOyBzaGFwZS1yZW5kZXJpbmc6IGF1dG87IiB2aWV3Qm94PSIwIDAgMTAwIDEwMCIgcHJlc2VydmVBc3BlY3RSYXRpbz0ieE1pZFlNaWQiPiAgPGNpcmNsZSBjeD0iNTAiIGN5PSI1MCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjYzNjM2MzIiBzdHJva2Utd2lkdGg9IjEwIiByPSIzNSIgc3Ryb2tlLWRhc2hhcnJheT0iMTY0LjkzMzYxNDMxMzQ2NDE1IDU2Ljk3Nzg3MTQzNzgyMTM4Ij4gICAgPGFuaW1hdGVUcmFuc2Zvcm0gYXR0cmlidXRlTmFtZT0idHJhbnNmb3JtIiB0eXBlPSJyb3RhdGUiIHJlcGVhdENvdW50PSJpbmRlZmluaXRlIiBkdXI9IjFzIiB2YWx1ZXM9IjAgNTAgNTA7MzYwIDUwIDUwIiBrZXlUaW1lcz0iMDsxIj48L2FuaW1hdGVUcmFuc2Zvcm0+ICA8L2NpcmNsZT48L3N2Zz4=\");\n  background-size: auto calc(min(50%, 400px));\n}",{"id":"p3597"},{"id":"p3595"},{"id":"p3596"}],"margin":0,"sizing_mode":"stretch_width","align":"start"}},{"type":"object","name":"Figure","id":"p3606","attributes":{"width":700,"height":300,"margin":[5,10],"sizing_mode":"fixed","align":"start","x_range":{"type":"object","name":"Range1d","id":"p3599","attributes":{"tags":[[["k","k",null]],[]],"start":1.0,"end":10.0,"reset_start":1.0,"reset_end":10.0}},"y_range":{"type":"object","name":"Range1d","id":"p3600","attributes":{"tags":[[["inertia","inertia",null]],{"type":"map","entries":[["invert_yaxis",false],["autorange",false]]}],"start":2.1827779456887484,"end":312.89247473221013,"reset_start":2.1827779456887484,"reset_end":312.89247473221013}},"x_scale":{"type":"object","name":"LinearScale","id":"p3616"},"y_scale":{"type":"object","name":"LinearScale","id":"p3617"},"title":{"type":"object","name":"Title","id":"p3609","attributes":{"text":"Elbow Curve","text_color":"black","text_font_size":"12pt"}},"renderers":[{"type":"object","name":"GlyphRenderer","id":"p3642","attributes":{"data_source":{"type":"object","name":"ColumnDataSource","id":"p3633","attributes":{"selected":{"type":"object","name":"Selection","id":"p3634","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"p3635"},"data":{"type":"map","entries":[["k",{"type":"ndarray","array":{"type":"bytes","data":"AQAAAAIAAAADAAAABAAAAAUAAAAGAAAABwAAAAgAAAAJAAAACgAAAA=="},"shape":[10],"dtype":"int32","order":"little"}],["inertia",{"type":"ndarray","array":{"type":"bytes","data":"AAAAAADwcUDlwndVTNJoQOCRvNowzF5AGO6ulG/BU0BnEQsuWlNQQOdykvoHDUpAmOLlEoMER0DEjt+KdKRCQIIPyVCpFEBAFoNuwkMTPEA="},"shape":[10],"dtype":"float64","order":"little"}]]}}},"view":{"type":"object","name":"CDSView","id":"p3643","attributes":{"filter":{"type":"object","name":"AllIndices","id":"p3644"}}},"glyph":{"type":"object","name":"Line","id":"p3639","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"k"},"y":{"type":"field","field":"inertia"},"line_color":"#30a2da","line_width":2}},"selection_glyph":{"type":"object","name":"Line","id":"p3645","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"k"},"y":{"type":"field","field":"inertia"},"line_color":"#30a2da","line_width":2}},"nonselection_glyph":{"type":"object","name":"Line","id":"p3640","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"k"},"y":{"type":"field","field":"inertia"},"line_color":"#30a2da","line_alpha":0.1,"line_width":2}},"muted_glyph":{"type":"object","name":"Line","id":"p3641","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"k"},"y":{"type":"field","field":"inertia"},"line_color":"#30a2da","line_alpha":0.2,"line_width":2}}}}],"toolbar":{"type":"object","name":"Toolbar","id":"p3615","attributes":{"tools":[{"type":"object","name":"WheelZoomTool","id":"p3604","attributes":{"tags":["hv_created"],"zoom_together":"none"}},{"type":"object","name":"HoverTool","id":"p3605","attributes":{"tags":["hv_created"],"renderers":[{"id":"p3642"}],"tooltips":[["k","@{k}"],["inertia","@{inertia}"]]}},{"type":"object","name":"SaveTool","id":"p3628"},{"type":"object","name":"PanTool","id":"p3629"},{"type":"object","name":"BoxZoomTool","id":"p3630","attributes":{"overlay":{"type":"object","name":"BoxAnnotation","id":"p3631","attributes":{"syncable":false,"level":"overlay","visible":false,"left_units":"canvas","right_units":"canvas","bottom_units":"canvas","top_units":"canvas","line_color":"black","line_alpha":1.0,"line_width":2,"line_dash":[4,4],"fill_color":"lightgrey","fill_alpha":0.5}}}},{"type":"object","name":"ResetTool","id":"p3632"}],"active_drag":{"id":"p3629"},"active_scroll":{"id":"p3604"}}},"left":[{"type":"object","name":"LinearAxis","id":"p3623","attributes":{"ticker":{"type":"object","name":"BasicTicker","id":"p3624","attributes":{"mantissas":[1,2,5]}},"formatter":{"type":"object","name":"BasicTickFormatter","id":"p3625"},"axis_label":"inertia","major_label_policy":{"type":"object","name":"AllLabels","id":"p3626"}}}],"below":[{"type":"object","name":"LinearAxis","id":"p3618","attributes":{"ticker":{"type":"object","name":"FixedTicker","id":"p3646","attributes":{"ticks":[1,2,3,4,5,6,7,8,9,10],"minor_ticks":[]}},"formatter":{"type":"object","name":"BasicTickFormatter","id":"p3620"},"axis_label":"k","major_label_policy":{"type":"object","name":"AllLabels","id":"p3621"}}}],"center":[{"type":"object","name":"Grid","id":"p3622","attributes":{"axis":{"id":"p3618"},"grid_line_color":null}},{"type":"object","name":"Grid","id":"p3627","attributes":{"dimension":1,"axis":{"id":"p3623"},"grid_line_color":null}}],"min_border_top":10,"min_border_bottom":10,"min_border_left":10,"min_border_right":10,"output_backend":"webgl"}},{"type":"object","name":"Spacer","id":"p3648","attributes":{"name":"HSpacer07084","stylesheets":["\n:host(.pn-loading.pn-arc):before, .pn-loading.pn-arc:before {\n  background-image: url(\"data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHN0eWxlPSJtYXJnaW46IGF1dG87IGJhY2tncm91bmQ6IG5vbmU7IGRpc3BsYXk6IGJsb2NrOyBzaGFwZS1yZW5kZXJpbmc6IGF1dG87IiB2aWV3Qm94PSIwIDAgMTAwIDEwMCIgcHJlc2VydmVBc3BlY3RSYXRpbz0ieE1pZFlNaWQiPiAgPGNpcmNsZSBjeD0iNTAiIGN5PSI1MCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjYzNjM2MzIiBzdHJva2Utd2lkdGg9IjEwIiByPSIzNSIgc3Ryb2tlLWRhc2hhcnJheT0iMTY0LjkzMzYxNDMxMzQ2NDE1IDU2Ljk3Nzg3MTQzNzgyMTM4Ij4gICAgPGFuaW1hdGVUcmFuc2Zvcm0gYXR0cmlidXRlTmFtZT0idHJhbnNmb3JtIiB0eXBlPSJyb3RhdGUiIHJlcGVhdENvdW50PSJpbmRlZmluaXRlIiBkdXI9IjFzIiB2YWx1ZXM9IjAgNTAgNTA7MzYwIDUwIDUwIiBrZXlUaW1lcz0iMDsxIj48L2FuaW1hdGVUcmFuc2Zvcm0+ICA8L2NpcmNsZT48L3N2Zz4=\");\n  background-size: auto calc(min(50%, 400px));\n}",{"id":"p3597"},{"id":"p3595"},{"id":"p3596"}],"margin":0,"sizing_mode":"stretch_width","align":"start"}}]}}],"defs":[{"type":"model","name":"ReactiveHTML1"},{"type":"model","name":"FlexBox1","properties":[{"name":"align_content","kind":"Any","default":"flex-start"},{"name":"align_items","kind":"Any","default":"flex-start"},{"name":"flex_direction","kind":"Any","default":"row"},{"name":"flex_wrap","kind":"Any","default":"wrap"},{"name":"justify_content","kind":"Any","default":"flex-start"}]},{"type":"model","name":"FloatPanel1","properties":[{"name":"config","kind":"Any","default":{"type":"map"}},{"name":"contained","kind":"Any","default":true},{"name":"position","kind":"Any","default":"right-top"},{"name":"offsetx","kind":"Any","default":null},{"name":"offsety","kind":"Any","default":null},{"name":"theme","kind":"Any","default":"primary"},{"name":"status","kind":"Any","default":"normalized"}]},{"type":"model","name":"GridStack1","properties":[{"name":"mode","kind":"Any","default":"warn"},{"name":"ncols","kind":"Any","default":null},{"name":"nrows","kind":"Any","default":null},{"name":"allow_resize","kind":"Any","default":true},{"name":"allow_drag","kind":"Any","default":true},{"name":"state","kind":"Any","default":[]}]},{"type":"model","name":"drag1","properties":[{"name":"slider_width","kind":"Any","default":5},{"name":"slider_color","kind":"Any","default":"black"},{"name":"value","kind":"Any","default":50}]},{"type":"model","name":"click1","properties":[{"name":"terminal_output","kind":"Any","default":""},{"name":"debug_name","kind":"Any","default":""},{"name":"clears","kind":"Any","default":0}]},{"type":"model","name":"FastWrapper1","properties":[{"name":"object","kind":"Any","default":null},{"name":"style","kind":"Any","default":null}]},{"type":"model","name":"NotificationAreaBase1","properties":[{"name":"js_events","kind":"Any","default":{"type":"map"}},{"name":"position","kind":"Any","default":"bottom-right"},{"name":"_clear","kind":"Any","default":0}]},{"type":"model","name":"NotificationArea1","properties":[{"name":"js_events","kind":"Any","default":{"type":"map"}},{"name":"notifications","kind":"Any","default":[]},{"name":"position","kind":"Any","default":"bottom-right"},{"name":"_clear","kind":"Any","default":0},{"name":"types","kind":"Any","default":[{"type":"map","entries":[["type","warning"],["background","#ffc107"],["icon",{"type":"map","entries":[["className","fas fa-exclamation-triangle"],["tagName","i"],["color","white"]]}]]},{"type":"map","entries":[["type","info"],["background","#007bff"],["icon",{"type":"map","entries":[["className","fas fa-info-circle"],["tagName","i"],["color","white"]]}]]}]}]},{"type":"model","name":"Notification","properties":[{"name":"background","kind":"Any","default":null},{"name":"duration","kind":"Any","default":3000},{"name":"icon","kind":"Any","default":null},{"name":"message","kind":"Any","default":""},{"name":"notification_type","kind":"Any","default":null},{"name":"_destroyed","kind":"Any","default":false}]},{"type":"model","name":"TemplateActions1","properties":[{"name":"open_modal","kind":"Any","default":0},{"name":"close_modal","kind":"Any","default":0}]},{"type":"model","name":"BootstrapTemplateActions1","properties":[{"name":"open_modal","kind":"Any","default":0},{"name":"close_modal","kind":"Any","default":0}]},{"type":"model","name":"MaterialTemplateActions1","properties":[{"name":"open_modal","kind":"Any","default":0},{"name":"close_modal","kind":"Any","default":0}]}]}};
  var render_items = [{"docid":"d19d7030-1171-4339-8dc1-d613a1d576a4","roots":{"p3594":"fed44dbe-8ef3-40cc-ae73-b13d0b69db86"},"root_ids":["p3594"]}];
  var docs = Object.values(docs_json)
  if (!docs) {
    return
  }
  const py_version = docs[0].version.replace('rc', '-rc.').replace('.dev', '-dev.')
  const is_dev = py_version.indexOf("+") !== -1 || py_version.indexOf("-") !== -1
  function embed_document(root) {
    var Bokeh = get_bokeh(root)
    Bokeh.embed.embed_items_notebook(docs_json, render_items);
    for (const render_item of render_items) {
      for (const root_id of render_item.root_ids) {
	const id_el = document.getElementById(root_id)
	if (id_el.children.length && (id_el.children[0].className === 'bk-root')) {
	  const root_el = id_el.children[0]
	  root_el.id = root_el.id + '-rendered'
	}
      }
    }
  }
  function get_bokeh(root) {
    if (root.Bokeh === undefined) {
      return null
    } else if (root.Bokeh.version !== py_version && !is_dev) {
      if (root.Bokeh.versions === undefined || !root.Bokeh.versions.has(py_version)) {
	return null
      }
      return root.Bokeh.versions.get(py_version);
    } else if (root.Bokeh.version === py_version) {
      return root.Bokeh
    }
    return null
  }
  function is_loaded(root) {
    var Bokeh = get_bokeh(root)
    return (Bokeh != null && Bokeh.Panel !== undefined)
  }
  if (is_loaded(root)) {
    embed_document(root);
  } else {
    var attempts = 0;
    var timer = setInterval(function(root) {
      if (is_loaded(root)) {
        clearInterval(timer);
        embed_document(root);
      } else if (document.readyState == "complete") {
        attempts++;
        if (attempts > 200) {
          clearInterval(timer);
	  var Bokeh = get_bokeh(root)
	  if (Bokeh == null || Bokeh.Panel == null) {
            console.warn("Panel: ERROR: Unable to run Panel code because Bokeh or Panel library is missing");
	  } else {
	    console.warn("Panel: WARNING: Attempting to render but not all required libraries could be resolved.")
	    embed_document(root)
	  }
        }
      }
    }, 25, root)
  }
})(window);</script>




```python
Answer the following question: What is the best value for k?
Question: What is the best value for k?
Answer: 4
```


    [1;31mType:[0m        list
    [1;31mString form:[0m [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    [1;31mLength:[0m      10
    [1;31mDocstring:[0m  
    Built-in mutable sequence.
    
    If no argument is given, the constructor creates a new empty list.
    The argument must be an iterable if specified.


---

### Cluster Cryptocurrencies with K-means Using the Original Data

In this section, you will use the K-Means algorithm with the best value for `k` found in the previous section to cluster the cryptocurrencies according to the price changes of cryptocurrencies provided.

1. Initialize the K-Means model with four clusters using the best value for `k`. 

2. Fit the K-Means model using the original data.

3. Predict the clusters to group the cryptocurrencies using the original data. View the resulting array of cluster values.

4. Create a copy of the original data and add a new column with the predicted clusters.

5. Create a scatter plot using hvPlot by setting `x="price_change_percentage_24h"` and `y="price_change_percentage_7d"`. Color the graph points with the labels found using K-Means and add the crypto name in the `hover_cols` parameter to identify the cryptocurrency represented by each data point.


```python
# Initialize the K-Means model using the best value for k
model = KMeans(n_clusters=4)
```


```python
# Fit the K-Means model using the scaled data
model.fit(df_market_data_scaled)
```

    C:\Users\HPs\anaconda3\Lib\site-packages\sklearn\cluster\_kmeans.py:1412: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      super()._check_params_vs_input(X, default_n_init=10)
    C:\Users\HPs\anaconda3\Lib\site-packages\sklearn\cluster\_kmeans.py:1436: UserWarning: KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS=1.
      warnings.warn(
    




<style>#sk-container-id-5 {color: black;}#sk-container-id-5 pre{padding: 0;}#sk-container-id-5 div.sk-toggleable {background-color: white;}#sk-container-id-5 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-5 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-5 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-5 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-5 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-5 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-5 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-5 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-5 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-5 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-5 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-5 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-5 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-5 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-5 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-5 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-5 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-5 div.sk-item {position: relative;z-index: 1;}#sk-container-id-5 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-5 div.sk-item::before, #sk-container-id-5 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-5 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-5 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-5 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-5 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-5 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-5 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-5 div.sk-label-container {text-align: center;}#sk-container-id-5 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-5 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-5" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>KMeans(n_clusters=4)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-5" type="checkbox" checked><label for="sk-estimator-id-5" class="sk-toggleable__label sk-toggleable__label-arrow">KMeans</label><div class="sk-toggleable__content"><pre>KMeans(n_clusters=4)</pre></div></div></div></div></div>




```python
# Predict the clusters to group the cryptocurrencies using the scaled data
data_predict = model.predict(df_market_data_scaled)
# View the resulting array of cluster values.
data_predict
```




    array([1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1,
           0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 1, 0, 0, 3, 0, 0, 0, 0])




```python
# Create a copy of the DataFrame
df_market_data_scaled.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price_change_percentage_24h</th>
      <th>price_change_percentage_7d</th>
      <th>price_change_percentage_14d</th>
      <th>price_change_percentage_30d</th>
      <th>price_change_percentage_60d</th>
      <th>price_change_percentage_200d</th>
      <th>price_change_percentage_1y</th>
    </tr>
    <tr>
      <th>coin_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>bitcoin</th>
      <td>0.508529</td>
      <td>0.493193</td>
      <td>0.772200</td>
      <td>0.235460</td>
      <td>-0.067495</td>
      <td>-0.355953</td>
      <td>-0.251637</td>
    </tr>
    <tr>
      <th>ethereum</th>
      <td>0.185446</td>
      <td>0.934445</td>
      <td>0.558692</td>
      <td>-0.054341</td>
      <td>-0.273483</td>
      <td>-0.115759</td>
      <td>-0.199352</td>
    </tr>
    <tr>
      <th>tether</th>
      <td>0.021774</td>
      <td>-0.706337</td>
      <td>-0.021680</td>
      <td>-0.061030</td>
      <td>0.008005</td>
      <td>-0.550247</td>
      <td>-0.282061</td>
    </tr>
    <tr>
      <th>ripple</th>
      <td>-0.040764</td>
      <td>-0.810928</td>
      <td>0.249458</td>
      <td>-0.050388</td>
      <td>-0.373164</td>
      <td>-0.458259</td>
      <td>-0.295546</td>
    </tr>
    <tr>
      <th>bitcoin-cash</th>
      <td>1.193036</td>
      <td>2.000959</td>
      <td>1.760610</td>
      <td>0.545842</td>
      <td>-0.291203</td>
      <td>-0.499848</td>
      <td>-0.270317</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Add a new column to the DataFrame with the predicted clusters
df_market_data_scaled["Predicted Clusters"] = data_predict
                      
# Display sample data
df_market_data_scaled.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price_change_percentage_24h</th>
      <th>price_change_percentage_7d</th>
      <th>price_change_percentage_14d</th>
      <th>price_change_percentage_30d</th>
      <th>price_change_percentage_60d</th>
      <th>price_change_percentage_200d</th>
      <th>price_change_percentage_1y</th>
      <th>Predicted Clusters</th>
    </tr>
    <tr>
      <th>coin_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>bitcoin</th>
      <td>0.508529</td>
      <td>0.493193</td>
      <td>0.772200</td>
      <td>0.235460</td>
      <td>-0.067495</td>
      <td>-0.355953</td>
      <td>-0.251637</td>
      <td>1</td>
    </tr>
    <tr>
      <th>ethereum</th>
      <td>0.185446</td>
      <td>0.934445</td>
      <td>0.558692</td>
      <td>-0.054341</td>
      <td>-0.273483</td>
      <td>-0.115759</td>
      <td>-0.199352</td>
      <td>1</td>
    </tr>
    <tr>
      <th>tether</th>
      <td>0.021774</td>
      <td>-0.706337</td>
      <td>-0.021680</td>
      <td>-0.061030</td>
      <td>0.008005</td>
      <td>-0.550247</td>
      <td>-0.282061</td>
      <td>0</td>
    </tr>
    <tr>
      <th>ripple</th>
      <td>-0.040764</td>
      <td>-0.810928</td>
      <td>0.249458</td>
      <td>-0.050388</td>
      <td>-0.373164</td>
      <td>-0.458259</td>
      <td>-0.295546</td>
      <td>0</td>
    </tr>
    <tr>
      <th>bitcoin-cash</th>
      <td>1.193036</td>
      <td>2.000959</td>
      <td>1.760610</td>
      <td>0.545842</td>
      <td>-0.291203</td>
      <td>-0.499848</td>
      <td>-0.270317</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Create a scatter plot using hvPlot by setting 
# `x="price_change_percentage_24h"` and `y="price_change_percentage_7d"`. 
# Color the graph points with the labels found using K-Means and 
# add the crypto name in the `hover_cols` parameter to identify 
# the cryptocurrency represented by each data point.
df_market_data_scaled.hvplot.scatter(
    x = "price_change_percentage_24h",
    y = "price_change_percentage_7d",
    by = "Predicted Clusters",
    hover_cols = "coin_id",
   title = "Segement Clusters",
    xlabel = "Price Change Percentage 24 Hours",
    ylabel = "Price Change Percentage 7 Days",    
)
```






<div id='p3654'>
  <div id="bbee631e-fc01-4110-b9d3-3d61fb4c8f00" data-root-id="p3654" style="display: contents;"></div>
</div>
<script type="application/javascript">(function(root) {
  var docs_json = {"2d2bf62c-cb36-41cf-9a0d-3ad5ac4a75ec":{"version":"3.2.1","title":"Bokeh Application","roots":[{"type":"object","name":"Row","id":"p3654","attributes":{"name":"Row07266","tags":["embedded"],"stylesheets":["\n:host(.pn-loading.pn-arc):before, .pn-loading.pn-arc:before {\n  background-image: url(\"data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHN0eWxlPSJtYXJnaW46IGF1dG87IGJhY2tncm91bmQ6IG5vbmU7IGRpc3BsYXk6IGJsb2NrOyBzaGFwZS1yZW5kZXJpbmc6IGF1dG87IiB2aWV3Qm94PSIwIDAgMTAwIDEwMCIgcHJlc2VydmVBc3BlY3RSYXRpbz0ieE1pZFlNaWQiPiAgPGNpcmNsZSBjeD0iNTAiIGN5PSI1MCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjYzNjM2MzIiBzdHJva2Utd2lkdGg9IjEwIiByPSIzNSIgc3Ryb2tlLWRhc2hhcnJheT0iMTY0LjkzMzYxNDMxMzQ2NDE1IDU2Ljk3Nzg3MTQzNzgyMTM4Ij4gICAgPGFuaW1hdGVUcmFuc2Zvcm0gYXR0cmlidXRlTmFtZT0idHJhbnNmb3JtIiB0eXBlPSJyb3RhdGUiIHJlcGVhdENvdW50PSJpbmRlZmluaXRlIiBkdXI9IjFzIiB2YWx1ZXM9IjAgNTAgNTA7MzYwIDUwIDUwIiBrZXlUaW1lcz0iMDsxIj48L2FuaW1hdGVUcmFuc2Zvcm0+ICA8L2NpcmNsZT48L3N2Zz4=\");\n  background-size: auto calc(min(50%, 400px));\n}",{"type":"object","name":"ImportedStyleSheet","id":"p3657","attributes":{"url":"https://cdn.holoviz.org/panel/1.2.1/dist/css/loading.css"}},{"type":"object","name":"ImportedStyleSheet","id":"p3762","attributes":{"url":"https://cdn.holoviz.org/panel/1.2.1/dist/css/listpanel.css"}},{"type":"object","name":"ImportedStyleSheet","id":"p3655","attributes":{"url":"https://cdn.holoviz.org/panel/1.2.1/dist/bundled/theme/default.css"}},{"type":"object","name":"ImportedStyleSheet","id":"p3656","attributes":{"url":"https://cdn.holoviz.org/panel/1.2.1/dist/bundled/theme/native.css"}}],"min_width":700,"margin":0,"sizing_mode":"stretch_width","align":"start","children":[{"type":"object","name":"Spacer","id":"p3658","attributes":{"name":"HSpacer07277","stylesheets":["\n:host(.pn-loading.pn-arc):before, .pn-loading.pn-arc:before {\n  background-image: url(\"data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHN0eWxlPSJtYXJnaW46IGF1dG87IGJhY2tncm91bmQ6IG5vbmU7IGRpc3BsYXk6IGJsb2NrOyBzaGFwZS1yZW5kZXJpbmc6IGF1dG87IiB2aWV3Qm94PSIwIDAgMTAwIDEwMCIgcHJlc2VydmVBc3BlY3RSYXRpbz0ieE1pZFlNaWQiPiAgPGNpcmNsZSBjeD0iNTAiIGN5PSI1MCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjYzNjM2MzIiBzdHJva2Utd2lkdGg9IjEwIiByPSIzNSIgc3Ryb2tlLWRhc2hhcnJheT0iMTY0LjkzMzYxNDMxMzQ2NDE1IDU2Ljk3Nzg3MTQzNzgyMTM4Ij4gICAgPGFuaW1hdGVUcmFuc2Zvcm0gYXR0cmlidXRlTmFtZT0idHJhbnNmb3JtIiB0eXBlPSJyb3RhdGUiIHJlcGVhdENvdW50PSJpbmRlZmluaXRlIiBkdXI9IjFzIiB2YWx1ZXM9IjAgNTAgNTA7MzYwIDUwIDUwIiBrZXlUaW1lcz0iMDsxIj48L2FuaW1hdGVUcmFuc2Zvcm0+ICA8L2NpcmNsZT48L3N2Zz4=\");\n  background-size: auto calc(min(50%, 400px));\n}",{"id":"p3657"},{"id":"p3655"},{"id":"p3656"}],"margin":0,"sizing_mode":"stretch_width","align":"start"}},{"type":"object","name":"Figure","id":"p3675","attributes":{"width":700,"height":300,"margin":[5,10],"sizing_mode":"fixed","align":"start","x_range":{"type":"object","name":"Range1d","id":"p3659","attributes":{"tags":[[["price_change_percentage_24h","price_change_percentage_24h",null]],[]],"start":-5.276792781891412,"end":2.2155632386560065,"reset_start":-5.276792781891412,"reset_end":2.2155632386560065}},"y_range":{"type":"object","name":"Range1d","id":"p3660","attributes":{"tags":[[["price_change_percentage_7d","price_change_percentage_7d",null]],{"type":"map","entries":[["invert_yaxis",false],["autorange",false]]}],"start":-2.107454305728652,"end":2.997678656273595,"reset_start":-2.107454305728652,"reset_end":2.997678656273595}},"x_scale":{"type":"object","name":"LinearScale","id":"p3685"},"y_scale":{"type":"object","name":"LinearScale","id":"p3686"},"title":{"type":"object","name":"Title","id":"p3678","attributes":{"text":"Segement Clusters","text_color":"black","text_font_size":"12pt"}},"renderers":[{"type":"object","name":"GlyphRenderer","id":"p3711","attributes":{"name":"0","data_source":{"type":"object","name":"ColumnDataSource","id":"p3702","attributes":{"selected":{"type":"object","name":"Selection","id":"p3703","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"p3704"},"data":{"type":"map","entries":[["price_change_percentage_24h",{"type":"ndarray","array":{"type":"bytes","data":"mCJF1OlLlj/mJRG3Ed+kv1qZGRifu9w//rJXHrAz1T/yMTqenpahPx9kk3dK7sM/GTRehHalwD90fnl54Irtv9XPDahDw9G/+VOWQSEmxz+kpbukw1bSP6eM4M4OgKo/BcMOj9+jqr/Chq6S5ebLv2h3TIfCZ68/weEQ8QyV0D/KmZTmC7niP1YH9WGFy/m/N7JuVBEB07/EU4GAhEGyv8Ocf0hw2/a/lrPBJY23/j/IKcQzxy3av5WrmDyeito/Bf22eE/6sz9Y362Ir3rzPw=="},"shape":[26],"dtype":"float64","order":"little"}],["price_change_percentage_7d",{"type":"ndarray","array":{"type":"bytes","data":"bZSIvk+a5r9RujxqH/Ppvw2U7adXaMi/d+r7VGbW+b9EGSx/8nTnv4rfBo4Lhe2/BEBwXEEApb8Udj8VbP/2vySelkVDp9i/7SZvVsyO5r8uu+JVH+LUvziN+gRMte2/o2qlez1D3b/CtPLKIS7rvyYjaswHnea/US6joeTvzz/dRmuCvtDvv/9TV66U6fq/ZWX+mWVCuD8GJ0pHuV/Nv8nIE2ZWH5q/TTxj/me11z+Htdvf1gXtv2Avihmzf9o/e31xtwIC5r/STlu6Y3Ljvw=="},"shape":[26],"dtype":"float64","order":"little"}],["coin_id",["tether","ripple","bitcoin-cash-sv","crypto-com-chain","usd-coin","eos","tron","okb","stellar","cdai","neo","leo-token","huobi-token","nem","binance-usd","iota","vechain","theta-token","dash","ethereum-classic","havven","omisego","ontology","ftx-token","true-usd","digibyte"]],["Predicted_Clusters",[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]]}}},"view":{"type":"object","name":"CDSView","id":"p3712","attributes":{"filter":{"type":"object","name":"AllIndices","id":"p3713"}}},"glyph":{"type":"object","name":"Scatter","id":"p3708","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"price_change_percentage_24h"},"y":{"type":"field","field":"price_change_percentage_7d"},"size":{"type":"value","value":5.477225575051661},"line_color":{"type":"value","value":"#30a2da"},"fill_color":{"type":"value","value":"#30a2da"},"hatch_color":{"type":"value","value":"#30a2da"}}},"selection_glyph":{"type":"object","name":"Scatter","id":"p3716","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"price_change_percentage_24h"},"y":{"type":"field","field":"price_change_percentage_7d"},"size":{"type":"value","value":5.477225575051661},"angle":{"type":"value","value":0.0},"line_color":{"type":"value","value":"#30a2da"},"line_alpha":{"type":"value","value":1.0},"line_width":{"type":"value","value":1},"line_join":{"type":"value","value":"bevel"},"line_cap":{"type":"value","value":"butt"},"line_dash":{"type":"value","value":[]},"line_dash_offset":{"type":"value","value":0},"fill_color":{"type":"value","value":"#30a2da"},"fill_alpha":{"type":"value","value":1.0},"hatch_color":{"type":"value","value":"#30a2da"},"hatch_alpha":{"type":"value","value":1.0},"hatch_scale":{"type":"value","value":12.0},"hatch_pattern":{"type":"value","value":null},"hatch_weight":{"type":"value","value":1.0},"marker":{"type":"value","value":"circle"}}},"nonselection_glyph":{"type":"object","name":"Scatter","id":"p3709","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"price_change_percentage_24h"},"y":{"type":"field","field":"price_change_percentage_7d"},"size":{"type":"value","value":5.477225575051661},"line_color":{"type":"value","value":"#30a2da"},"line_alpha":{"type":"value","value":0.1},"fill_color":{"type":"value","value":"#30a2da"},"fill_alpha":{"type":"value","value":0.1},"hatch_color":{"type":"value","value":"#30a2da"},"hatch_alpha":{"type":"value","value":0.1}}},"muted_glyph":{"type":"object","name":"Scatter","id":"p3710","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"price_change_percentage_24h"},"y":{"type":"field","field":"price_change_percentage_7d"},"size":{"type":"value","value":5.477225575051661},"line_color":{"type":"value","value":"#30a2da"},"line_alpha":{"type":"value","value":0.2},"fill_color":{"type":"value","value":"#30a2da"},"fill_alpha":{"type":"value","value":0.2},"hatch_color":{"type":"value","value":"#30a2da"},"hatch_alpha":{"type":"value","value":0.2}}}}},{"type":"object","name":"GlyphRenderer","id":"p3726","attributes":{"name":"1","data_source":{"type":"object","name":"ColumnDataSource","id":"p3717","attributes":{"selected":{"type":"object","name":"Selection","id":"p3718","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"p3719"},"data":{"type":"map","entries":[["price_change_percentage_24h",{"type":"ndarray","array":{"type":"bytes","data":"DdlqYN9F4D86azDosLzHPzALIwCtFvM/9w6McDSK7D/VDHOpb1eHPzMEvaFkP7o/Sds39trWsz9Epj8/c9DQP4xAeNEVZ8O/qq8st/Fh0L8FWWk3mH7gP6wo083aUMC//ACFOIMFwL8="},"shape":[13],"dtype":"float64","order":"little"}],["price_change_percentage_7d",{"type":"ndarray","array":{"type":"bytes","data":"U1k8q3mQ3z9yoPpI+ebtPzAB2dP2AQBAgmkOMZk89T8FNulI+JMEQCZRaSLGIPg/84wjaB1l1T8rMp3jf678P2VmURqKqeY/XK6ZksNx/T9DfMqy1o7dPyTPDB1Xu+0/srW7TleV4j8="},"shape":[13],"dtype":"float64","order":"little"}],["coin_id",["bitcoin","ethereum","bitcoin-cash","binancecoin","chainlink","cardano","litecoin","monero","tezos","cosmos","wrapped-bitcoin","zcash","maker"]],["Predicted_Clusters",[1,1,1,1,1,1,1,1,1,1,1,1,1]]]}}},"view":{"type":"object","name":"CDSView","id":"p3727","attributes":{"filter":{"type":"object","name":"AllIndices","id":"p3728"}}},"glyph":{"type":"object","name":"Scatter","id":"p3723","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"price_change_percentage_24h"},"y":{"type":"field","field":"price_change_percentage_7d"},"size":{"type":"value","value":5.477225575051661},"line_color":{"type":"value","value":"#fc4f30"},"fill_color":{"type":"value","value":"#fc4f30"},"hatch_color":{"type":"value","value":"#fc4f30"}}},"selection_glyph":{"type":"object","name":"Scatter","id":"p3730","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"price_change_percentage_24h"},"y":{"type":"field","field":"price_change_percentage_7d"},"size":{"type":"value","value":5.477225575051661},"angle":{"type":"value","value":0.0},"line_color":{"type":"value","value":"#fc4f30"},"line_alpha":{"type":"value","value":1.0},"line_width":{"type":"value","value":1},"line_join":{"type":"value","value":"bevel"},"line_cap":{"type":"value","value":"butt"},"line_dash":{"type":"value","value":[]},"line_dash_offset":{"type":"value","value":0},"fill_color":{"type":"value","value":"#fc4f30"},"fill_alpha":{"type":"value","value":1.0},"hatch_color":{"type":"value","value":"#fc4f30"},"hatch_alpha":{"type":"value","value":1.0},"hatch_scale":{"type":"value","value":12.0},"hatch_pattern":{"type":"value","value":null},"hatch_weight":{"type":"value","value":1.0},"marker":{"type":"value","value":"circle"}}},"nonselection_glyph":{"type":"object","name":"Scatter","id":"p3724","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"price_change_percentage_24h"},"y":{"type":"field","field":"price_change_percentage_7d"},"size":{"type":"value","value":5.477225575051661},"line_color":{"type":"value","value":"#fc4f30"},"line_alpha":{"type":"value","value":0.1},"fill_color":{"type":"value","value":"#fc4f30"},"fill_alpha":{"type":"value","value":0.1},"hatch_color":{"type":"value","value":"#fc4f30"},"hatch_alpha":{"type":"value","value":0.1}}},"muted_glyph":{"type":"object","name":"Scatter","id":"p3725","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"price_change_percentage_24h"},"y":{"type":"field","field":"price_change_percentage_7d"},"size":{"type":"value","value":5.477225575051661},"line_color":{"type":"value","value":"#fc4f30"},"line_alpha":{"type":"value","value":0.2},"fill_color":{"type":"value","value":"#fc4f30"},"fill_alpha":{"type":"value","value":0.2},"hatch_color":{"type":"value","value":"#fc4f30"},"hatch_alpha":{"type":"value","value":0.2}}}}},{"type":"object","name":"GlyphRenderer","id":"p3740","attributes":{"name":"2","data_source":{"type":"object","name":"ColumnDataSource","id":"p3731","attributes":{"selected":{"type":"object","name":"Selection","id":"p3732","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"p3733"},"data":{"type":"map","entries":[["price_change_percentage_24h",{"type":"ndarray","array":{"type":"bytes","data":"3WeHPpbsE8A="},"shape":[1],"dtype":"float64","order":"little"}],["price_change_percentage_7d",{"type":"ndarray","array":{"type":"bytes","data":"dIWi2pshp78="},"shape":[1],"dtype":"float64","order":"little"}],["coin_id",["ethlend"]],["Predicted_Clusters",[2]]]}}},"view":{"type":"object","name":"CDSView","id":"p3741","attributes":{"filter":{"type":"object","name":"AllIndices","id":"p3742"}}},"glyph":{"type":"object","name":"Scatter","id":"p3737","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"price_change_percentage_24h"},"y":{"type":"field","field":"price_change_percentage_7d"},"size":{"type":"value","value":5.477225575051661},"line_color":{"type":"value","value":"#e5ae38"},"fill_color":{"type":"value","value":"#e5ae38"},"hatch_color":{"type":"value","value":"#e5ae38"}}},"selection_glyph":{"type":"object","name":"Scatter","id":"p3744","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"price_change_percentage_24h"},"y":{"type":"field","field":"price_change_percentage_7d"},"size":{"type":"value","value":5.477225575051661},"angle":{"type":"value","value":0.0},"line_color":{"type":"value","value":"#e5ae38"},"line_alpha":{"type":"value","value":1.0},"line_width":{"type":"value","value":1},"line_join":{"type":"value","value":"bevel"},"line_cap":{"type":"value","value":"butt"},"line_dash":{"type":"value","value":[]},"line_dash_offset":{"type":"value","value":0},"fill_color":{"type":"value","value":"#e5ae38"},"fill_alpha":{"type":"value","value":1.0},"hatch_color":{"type":"value","value":"#e5ae38"},"hatch_alpha":{"type":"value","value":1.0},"hatch_scale":{"type":"value","value":12.0},"hatch_pattern":{"type":"value","value":null},"hatch_weight":{"type":"value","value":1.0},"marker":{"type":"value","value":"circle"}}},"nonselection_glyph":{"type":"object","name":"Scatter","id":"p3738","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"price_change_percentage_24h"},"y":{"type":"field","field":"price_change_percentage_7d"},"size":{"type":"value","value":5.477225575051661},"line_color":{"type":"value","value":"#e5ae38"},"line_alpha":{"type":"value","value":0.1},"fill_color":{"type":"value","value":"#e5ae38"},"fill_alpha":{"type":"value","value":0.1},"hatch_color":{"type":"value","value":"#e5ae38"},"hatch_alpha":{"type":"value","value":0.1}}},"muted_glyph":{"type":"object","name":"Scatter","id":"p3739","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"price_change_percentage_24h"},"y":{"type":"field","field":"price_change_percentage_7d"},"size":{"type":"value","value":5.477225575051661},"line_color":{"type":"value","value":"#e5ae38"},"line_alpha":{"type":"value","value":0.2},"fill_color":{"type":"value","value":"#e5ae38"},"fill_alpha":{"type":"value","value":0.2},"hatch_color":{"type":"value","value":"#e5ae38"},"hatch_alpha":{"type":"value","value":0.2}}}}},{"type":"object","name":"GlyphRenderer","id":"p3754","attributes":{"name":"3","data_source":{"type":"object","name":"ColumnDataSource","id":"p3745","attributes":{"selected":{"type":"object","name":"Selection","id":"p3746","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"p3747"},"data":{"type":"map","entries":[["price_change_percentage_24h",{"type":"ndarray","array":{"type":"bytes","data":"VLqXBn668D8="},"shape":[1],"dtype":"float64","order":"little"}],["price_change_percentage_7d",{"type":"ndarray","array":{"type":"bytes","data":"Nv03JFjJ478="},"shape":[1],"dtype":"float64","order":"little"}],["coin_id",["celsius-degree-token"]],["Predicted_Clusters",[3]]]}}},"view":{"type":"object","name":"CDSView","id":"p3755","attributes":{"filter":{"type":"object","name":"AllIndices","id":"p3756"}}},"glyph":{"type":"object","name":"Scatter","id":"p3751","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"price_change_percentage_24h"},"y":{"type":"field","field":"price_change_percentage_7d"},"size":{"type":"value","value":5.477225575051661},"line_color":{"type":"value","value":"#6d904f"},"fill_color":{"type":"value","value":"#6d904f"},"hatch_color":{"type":"value","value":"#6d904f"}}},"selection_glyph":{"type":"object","name":"Scatter","id":"p3758","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"price_change_percentage_24h"},"y":{"type":"field","field":"price_change_percentage_7d"},"size":{"type":"value","value":5.477225575051661},"angle":{"type":"value","value":0.0},"line_color":{"type":"value","value":"#6d904f"},"line_alpha":{"type":"value","value":1.0},"line_width":{"type":"value","value":1},"line_join":{"type":"value","value":"bevel"},"line_cap":{"type":"value","value":"butt"},"line_dash":{"type":"value","value":[]},"line_dash_offset":{"type":"value","value":0},"fill_color":{"type":"value","value":"#6d904f"},"fill_alpha":{"type":"value","value":1.0},"hatch_color":{"type":"value","value":"#6d904f"},"hatch_alpha":{"type":"value","value":1.0},"hatch_scale":{"type":"value","value":12.0},"hatch_pattern":{"type":"value","value":null},"hatch_weight":{"type":"value","value":1.0},"marker":{"type":"value","value":"circle"}}},"nonselection_glyph":{"type":"object","name":"Scatter","id":"p3752","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"price_change_percentage_24h"},"y":{"type":"field","field":"price_change_percentage_7d"},"size":{"type":"value","value":5.477225575051661},"line_color":{"type":"value","value":"#6d904f"},"line_alpha":{"type":"value","value":0.1},"fill_color":{"type":"value","value":"#6d904f"},"fill_alpha":{"type":"value","value":0.1},"hatch_color":{"type":"value","value":"#6d904f"},"hatch_alpha":{"type":"value","value":0.1}}},"muted_glyph":{"type":"object","name":"Scatter","id":"p3753","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"price_change_percentage_24h"},"y":{"type":"field","field":"price_change_percentage_7d"},"size":{"type":"value","value":5.477225575051661},"line_color":{"type":"value","value":"#6d904f"},"line_alpha":{"type":"value","value":0.2},"fill_color":{"type":"value","value":"#6d904f"},"fill_alpha":{"type":"value","value":0.2},"hatch_color":{"type":"value","value":"#6d904f"},"hatch_alpha":{"type":"value","value":0.2}}}}}],"toolbar":{"type":"object","name":"Toolbar","id":"p3684","attributes":{"tools":[{"type":"object","name":"WheelZoomTool","id":"p3664","attributes":{"tags":["hv_created"],"zoom_together":"none"}},{"type":"object","name":"HoverTool","id":"p3665","attributes":{"tags":["hv_created"],"renderers":[{"id":"p3711"},{"id":"p3726"},{"id":"p3740"},{"id":"p3754"}],"tooltips":[["Predicted Clusters","@{Predicted_Clusters}"],["price_change_percentage_24h","@{price_change_percentage_24h}"],["price_change_percentage_7d","@{price_change_percentage_7d}"],["coin_id","@{coin_id}"]]}},{"type":"object","name":"SaveTool","id":"p3697"},{"type":"object","name":"PanTool","id":"p3698"},{"type":"object","name":"BoxZoomTool","id":"p3699","attributes":{"overlay":{"type":"object","name":"BoxAnnotation","id":"p3700","attributes":{"syncable":false,"level":"overlay","visible":false,"left_units":"canvas","right_units":"canvas","bottom_units":"canvas","top_units":"canvas","line_color":"black","line_alpha":1.0,"line_width":2,"line_dash":[4,4],"fill_color":"lightgrey","fill_alpha":0.5}}}},{"type":"object","name":"ResetTool","id":"p3701"}],"active_drag":{"id":"p3698"},"active_scroll":{"id":"p3664"}}},"left":[{"type":"object","name":"LinearAxis","id":"p3692","attributes":{"ticker":{"type":"object","name":"BasicTicker","id":"p3693","attributes":{"mantissas":[1,2,5]}},"formatter":{"type":"object","name":"BasicTickFormatter","id":"p3694"},"axis_label":"Price Change Percentage 7 Days","major_label_policy":{"type":"object","name":"AllLabels","id":"p3695"}}}],"right":[{"type":"object","name":"Legend","id":"p3714","attributes":{"location":[0,0],"title":"Predicted Clusters","click_policy":"mute","items":[{"type":"object","name":"LegendItem","id":"p3715","attributes":{"label":{"type":"value","value":"0"},"renderers":[{"id":"p3711"}]}},{"type":"object","name":"LegendItem","id":"p3729","attributes":{"label":{"type":"value","value":"1"},"renderers":[{"id":"p3726"}]}},{"type":"object","name":"LegendItem","id":"p3743","attributes":{"label":{"type":"value","value":"2"},"renderers":[{"id":"p3740"}]}},{"type":"object","name":"LegendItem","id":"p3757","attributes":{"label":{"type":"value","value":"3"},"renderers":[{"id":"p3754"}]}}]}}],"below":[{"type":"object","name":"LinearAxis","id":"p3687","attributes":{"ticker":{"type":"object","name":"BasicTicker","id":"p3688","attributes":{"mantissas":[1,2,5]}},"formatter":{"type":"object","name":"BasicTickFormatter","id":"p3689"},"axis_label":"Price Change Percentage 24 Hours","major_label_policy":{"type":"object","name":"AllLabels","id":"p3690"}}}],"center":[{"type":"object","name":"Grid","id":"p3691","attributes":{"axis":{"id":"p3687"},"grid_line_color":null}},{"type":"object","name":"Grid","id":"p3696","attributes":{"dimension":1,"axis":{"id":"p3692"},"grid_line_color":null}}],"min_border_top":10,"min_border_bottom":10,"min_border_left":10,"min_border_right":10,"output_backend":"webgl"}},{"type":"object","name":"Spacer","id":"p3760","attributes":{"name":"HSpacer07280","stylesheets":["\n:host(.pn-loading.pn-arc):before, .pn-loading.pn-arc:before {\n  background-image: url(\"data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHN0eWxlPSJtYXJnaW46IGF1dG87IGJhY2tncm91bmQ6IG5vbmU7IGRpc3BsYXk6IGJsb2NrOyBzaGFwZS1yZW5kZXJpbmc6IGF1dG87IiB2aWV3Qm94PSIwIDAgMTAwIDEwMCIgcHJlc2VydmVBc3BlY3RSYXRpbz0ieE1pZFlNaWQiPiAgPGNpcmNsZSBjeD0iNTAiIGN5PSI1MCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjYzNjM2MzIiBzdHJva2Utd2lkdGg9IjEwIiByPSIzNSIgc3Ryb2tlLWRhc2hhcnJheT0iMTY0LjkzMzYxNDMxMzQ2NDE1IDU2Ljk3Nzg3MTQzNzgyMTM4Ij4gICAgPGFuaW1hdGVUcmFuc2Zvcm0gYXR0cmlidXRlTmFtZT0idHJhbnNmb3JtIiB0eXBlPSJyb3RhdGUiIHJlcGVhdENvdW50PSJpbmRlZmluaXRlIiBkdXI9IjFzIiB2YWx1ZXM9IjAgNTAgNTA7MzYwIDUwIDUwIiBrZXlUaW1lcz0iMDsxIj48L2FuaW1hdGVUcmFuc2Zvcm0+ICA8L2NpcmNsZT48L3N2Zz4=\");\n  background-size: auto calc(min(50%, 400px));\n}",{"id":"p3657"},{"id":"p3655"},{"id":"p3656"}],"margin":0,"sizing_mode":"stretch_width","align":"start"}}]}}],"defs":[{"type":"model","name":"ReactiveHTML1"},{"type":"model","name":"FlexBox1","properties":[{"name":"align_content","kind":"Any","default":"flex-start"},{"name":"align_items","kind":"Any","default":"flex-start"},{"name":"flex_direction","kind":"Any","default":"row"},{"name":"flex_wrap","kind":"Any","default":"wrap"},{"name":"justify_content","kind":"Any","default":"flex-start"}]},{"type":"model","name":"FloatPanel1","properties":[{"name":"config","kind":"Any","default":{"type":"map"}},{"name":"contained","kind":"Any","default":true},{"name":"position","kind":"Any","default":"right-top"},{"name":"offsetx","kind":"Any","default":null},{"name":"offsety","kind":"Any","default":null},{"name":"theme","kind":"Any","default":"primary"},{"name":"status","kind":"Any","default":"normalized"}]},{"type":"model","name":"GridStack1","properties":[{"name":"mode","kind":"Any","default":"warn"},{"name":"ncols","kind":"Any","default":null},{"name":"nrows","kind":"Any","default":null},{"name":"allow_resize","kind":"Any","default":true},{"name":"allow_drag","kind":"Any","default":true},{"name":"state","kind":"Any","default":[]}]},{"type":"model","name":"drag1","properties":[{"name":"slider_width","kind":"Any","default":5},{"name":"slider_color","kind":"Any","default":"black"},{"name":"value","kind":"Any","default":50}]},{"type":"model","name":"click1","properties":[{"name":"terminal_output","kind":"Any","default":""},{"name":"debug_name","kind":"Any","default":""},{"name":"clears","kind":"Any","default":0}]},{"type":"model","name":"FastWrapper1","properties":[{"name":"object","kind":"Any","default":null},{"name":"style","kind":"Any","default":null}]},{"type":"model","name":"NotificationAreaBase1","properties":[{"name":"js_events","kind":"Any","default":{"type":"map"}},{"name":"position","kind":"Any","default":"bottom-right"},{"name":"_clear","kind":"Any","default":0}]},{"type":"model","name":"NotificationArea1","properties":[{"name":"js_events","kind":"Any","default":{"type":"map"}},{"name":"notifications","kind":"Any","default":[]},{"name":"position","kind":"Any","default":"bottom-right"},{"name":"_clear","kind":"Any","default":0},{"name":"types","kind":"Any","default":[{"type":"map","entries":[["type","warning"],["background","#ffc107"],["icon",{"type":"map","entries":[["className","fas fa-exclamation-triangle"],["tagName","i"],["color","white"]]}]]},{"type":"map","entries":[["type","info"],["background","#007bff"],["icon",{"type":"map","entries":[["className","fas fa-info-circle"],["tagName","i"],["color","white"]]}]]}]}]},{"type":"model","name":"Notification","properties":[{"name":"background","kind":"Any","default":null},{"name":"duration","kind":"Any","default":3000},{"name":"icon","kind":"Any","default":null},{"name":"message","kind":"Any","default":""},{"name":"notification_type","kind":"Any","default":null},{"name":"_destroyed","kind":"Any","default":false}]},{"type":"model","name":"TemplateActions1","properties":[{"name":"open_modal","kind":"Any","default":0},{"name":"close_modal","kind":"Any","default":0}]},{"type":"model","name":"BootstrapTemplateActions1","properties":[{"name":"open_modal","kind":"Any","default":0},{"name":"close_modal","kind":"Any","default":0}]},{"type":"model","name":"MaterialTemplateActions1","properties":[{"name":"open_modal","kind":"Any","default":0},{"name":"close_modal","kind":"Any","default":0}]}]}};
  var render_items = [{"docid":"2d2bf62c-cb36-41cf-9a0d-3ad5ac4a75ec","roots":{"p3654":"bbee631e-fc01-4110-b9d3-3d61fb4c8f00"},"root_ids":["p3654"]}];
  var docs = Object.values(docs_json)
  if (!docs) {
    return
  }
  const py_version = docs[0].version.replace('rc', '-rc.').replace('.dev', '-dev.')
  const is_dev = py_version.indexOf("+") !== -1 || py_version.indexOf("-") !== -1
  function embed_document(root) {
    var Bokeh = get_bokeh(root)
    Bokeh.embed.embed_items_notebook(docs_json, render_items);
    for (const render_item of render_items) {
      for (const root_id of render_item.root_ids) {
	const id_el = document.getElementById(root_id)
	if (id_el.children.length && (id_el.children[0].className === 'bk-root')) {
	  const root_el = id_el.children[0]
	  root_el.id = root_el.id + '-rendered'
	}
      }
    }
  }
  function get_bokeh(root) {
    if (root.Bokeh === undefined) {
      return null
    } else if (root.Bokeh.version !== py_version && !is_dev) {
      if (root.Bokeh.versions === undefined || !root.Bokeh.versions.has(py_version)) {
	return null
      }
      return root.Bokeh.versions.get(py_version);
    } else if (root.Bokeh.version === py_version) {
      return root.Bokeh
    }
    return null
  }
  function is_loaded(root) {
    var Bokeh = get_bokeh(root)
    return (Bokeh != null && Bokeh.Panel !== undefined)
  }
  if (is_loaded(root)) {
    embed_document(root);
  } else {
    var attempts = 0;
    var timer = setInterval(function(root) {
      if (is_loaded(root)) {
        clearInterval(timer);
        embed_document(root);
      } else if (document.readyState == "complete") {
        attempts++;
        if (attempts > 200) {
          clearInterval(timer);
	  var Bokeh = get_bokeh(root)
	  if (Bokeh == null || Bokeh.Panel == null) {
            console.warn("Panel: ERROR: Unable to run Panel code because Bokeh or Panel library is missing");
	  } else {
	    console.warn("Panel: WARNING: Attempting to render but not all required libraries could be resolved.")
	    embed_document(root)
	  }
        }
      }
    }, 25, root)
  }
})(window);</script>



---

### Optimize Clusters with Principal Component Analysis

In this section, you will perform a principal component analysis (PCA) and reduce the features to three principal components.

1. Create a PCA model instance and set `n_components=3`.

2. Use the PCA model to reduce to three principal components. View the first five rows of the DataFrame. 

3. Retrieve the explained variance to determine how much information can be attributed to each principal component.

4. Answer the following question: What is the total explained variance of the three principal components?

5. Create a new DataFrame with the PCA data. Be sure to set the `coin_id` index from the original DataFrame as the index for the new DataFrame. Review the resulting DataFrame.


```python
# Create a PCA model instance and set `n_components=3`.
pca = PCA(n_components = 3)
```


```python
# Use the PCA model with `fit_transform` to reduce to 
# three principal components.
df_market_pca = pca.fit_transform(df_market_data_scaled)

# View the first five rows of the DataFrame. 
df_market_pca[:5]
```




    array([[ 1.02716415e-01, -1.09294014e+00,  5.35184395e-01],
           [ 7.67117769e-02, -7.04779064e-01,  1.03685543e+00],
           [-6.19998579e-01, -1.97690890e-02, -6.54034410e-01],
           [-6.64724714e-01,  2.17397355e-03, -4.85221049e-01],
           [ 2.88738674e-01, -2.43700872e+00,  1.79915657e+00]])




```python
# Retrieve the explained variance to determine how much information 
# can be attributed to each principal component.
pca.explained_variance_ratio_
```




    array([0.38043081, 0.33516548, 0.17629339])



#### Answer the following question: What is the total explained variance of the three principal components?

**Question:** What is the total explained variance of the three principal components?

**Answer:** About 88%


```python
# Create a new DataFrame with the PCA data.
# Note: The code for this step is provided for you.

# Creating a DataFrame with the PCA data
df_scaled_market_pca = pd.DataFrame(
    df_market_pca,
    columns = ["price_change_percentage_24h", "price_change_percentage_7d", "price_change_percentage_14d"])

# Copy the crypto names from the original data
df_scaled_market_pca["coin_id"] = df_market_data.index

# Set the coinid column as index
df_scaled_market_pca = df_scaled_market_pca.set_index("coin_id")

# Display sample data
df_scaled_market_pca.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price_change_percentage_24h</th>
      <th>price_change_percentage_7d</th>
      <th>price_change_percentage_14d</th>
    </tr>
    <tr>
      <th>coin_id</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>bitcoin</th>
      <td>0.102716</td>
      <td>-1.092940</td>
      <td>0.535184</td>
    </tr>
    <tr>
      <th>ethereum</th>
      <td>0.076712</td>
      <td>-0.704779</td>
      <td>1.036855</td>
    </tr>
    <tr>
      <th>tether</th>
      <td>-0.619999</td>
      <td>-0.019769</td>
      <td>-0.654034</td>
    </tr>
    <tr>
      <th>ripple</th>
      <td>-0.664725</td>
      <td>0.002174</td>
      <td>-0.485221</td>
    </tr>
    <tr>
      <th>bitcoin-cash</th>
      <td>0.288739</td>
      <td>-2.437009</td>
      <td>1.799157</td>
    </tr>
  </tbody>
</table>
</div>



---

### Find the Best Value for k Using the PCA Data

In this section, you will use the elbow method to find the best value for `k` using the PCA data.

1. Code the elbow method algorithm and use the PCA data to find the best value for `k`. Use a range from 1 to 11. 

2. Plot a line chart with all the inertia values computed with the different values of `k` to visually identify the optimal value for `k`.

3. Answer the following questions: What is the best value for k when using the PCA data? Does it differ from the best k value found using the original data?


```python
# Create a list with the number of k-values to try
# Use a range from 1 to 11
k = list(range(1, 11))
```


```python
# Create an empy list to store the inertia values
inertia = []
```


```python
# Create a for loop to compute the inertia with each possible value of k
# Inside the loop:
# 1. Create a KMeans model using the loop counter for the n_clusters
# 2. Fit the model to the data using `df_market_data_pca`
# 3. Append the model.inertia_ to the inertia list
for i in k:
    model = KMeans(n_clusters = i, random_state = 0)
    model.fit(df_scaled_market_pca)
    inertia.append(model.inertia_)
```


```python
# Create a dictionary with the data to plot the Elbow curve
elbow_data_2 = {
    "k" : k,
    "inertia" : inertia
}

# Create a DataFrame with the data to plot the Elbow curve
elbow_data_2 = pd.DataFrame(elbow_data_2)
```


```python
elbow_data_2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>k</th>
      <th>inertia</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>272.113366</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>181.543059</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>99.671696</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>47.229370</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>36.070529</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>26.319251</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>19.998910</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>16.611514</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>13.079712</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>10.053568</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Plot a line chart with all the inertia values computed with 
# the different values of k to visually identify the optimal value for k.
elbow_data_2.hvplot.line(x = "k", y = "inertia", title = "Elbow Curve", xticks = k)
```






<div id='p3772'>
  <div id="f57cf9a0-4d67-4f57-8dc9-b1d56effefad" data-root-id="p3772" style="display: contents;"></div>
</div>
<script type="application/javascript">(function(root) {
  var docs_json = {"a764cdd6-cda6-472f-b960-edfd2205b32a":{"version":"3.2.1","title":"Bokeh Application","roots":[{"type":"object","name":"Row","id":"p3772","attributes":{"name":"Row07460","tags":["embedded"],"stylesheets":["\n:host(.pn-loading.pn-arc):before, .pn-loading.pn-arc:before {\n  background-image: url(\"data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHN0eWxlPSJtYXJnaW46IGF1dG87IGJhY2tncm91bmQ6IG5vbmU7IGRpc3BsYXk6IGJsb2NrOyBzaGFwZS1yZW5kZXJpbmc6IGF1dG87IiB2aWV3Qm94PSIwIDAgMTAwIDEwMCIgcHJlc2VydmVBc3BlY3RSYXRpbz0ieE1pZFlNaWQiPiAgPGNpcmNsZSBjeD0iNTAiIGN5PSI1MCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjYzNjM2MzIiBzdHJva2Utd2lkdGg9IjEwIiByPSIzNSIgc3Ryb2tlLWRhc2hhcnJheT0iMTY0LjkzMzYxNDMxMzQ2NDE1IDU2Ljk3Nzg3MTQzNzgyMTM4Ij4gICAgPGFuaW1hdGVUcmFuc2Zvcm0gYXR0cmlidXRlTmFtZT0idHJhbnNmb3JtIiB0eXBlPSJyb3RhdGUiIHJlcGVhdENvdW50PSJpbmRlZmluaXRlIiBkdXI9IjFzIiB2YWx1ZXM9IjAgNTAgNTA7MzYwIDUwIDUwIiBrZXlUaW1lcz0iMDsxIj48L2FuaW1hdGVUcmFuc2Zvcm0+ICA8L2NpcmNsZT48L3N2Zz4=\");\n  background-size: auto calc(min(50%, 400px));\n}",{"type":"object","name":"ImportedStyleSheet","id":"p3775","attributes":{"url":"https://cdn.holoviz.org/panel/1.2.1/dist/css/loading.css"}},{"type":"object","name":"ImportedStyleSheet","id":"p3828","attributes":{"url":"https://cdn.holoviz.org/panel/1.2.1/dist/css/listpanel.css"}},{"type":"object","name":"ImportedStyleSheet","id":"p3773","attributes":{"url":"https://cdn.holoviz.org/panel/1.2.1/dist/bundled/theme/default.css"}},{"type":"object","name":"ImportedStyleSheet","id":"p3774","attributes":{"url":"https://cdn.holoviz.org/panel/1.2.1/dist/bundled/theme/native.css"}}],"min_width":700,"margin":0,"sizing_mode":"stretch_width","align":"start","children":[{"type":"object","name":"Spacer","id":"p3776","attributes":{"name":"HSpacer07471","stylesheets":["\n:host(.pn-loading.pn-arc):before, .pn-loading.pn-arc:before {\n  background-image: url(\"data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHN0eWxlPSJtYXJnaW46IGF1dG87IGJhY2tncm91bmQ6IG5vbmU7IGRpc3BsYXk6IGJsb2NrOyBzaGFwZS1yZW5kZXJpbmc6IGF1dG87IiB2aWV3Qm94PSIwIDAgMTAwIDEwMCIgcHJlc2VydmVBc3BlY3RSYXRpbz0ieE1pZFlNaWQiPiAgPGNpcmNsZSBjeD0iNTAiIGN5PSI1MCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjYzNjM2MzIiBzdHJva2Utd2lkdGg9IjEwIiByPSIzNSIgc3Ryb2tlLWRhc2hhcnJheT0iMTY0LjkzMzYxNDMxMzQ2NDE1IDU2Ljk3Nzg3MTQzNzgyMTM4Ij4gICAgPGFuaW1hdGVUcmFuc2Zvcm0gYXR0cmlidXRlTmFtZT0idHJhbnNmb3JtIiB0eXBlPSJyb3RhdGUiIHJlcGVhdENvdW50PSJpbmRlZmluaXRlIiBkdXI9IjFzIiB2YWx1ZXM9IjAgNTAgNTA7MzYwIDUwIDUwIiBrZXlUaW1lcz0iMDsxIj48L2FuaW1hdGVUcmFuc2Zvcm0+ICA8L2NpcmNsZT48L3N2Zz4=\");\n  background-size: auto calc(min(50%, 400px));\n}",{"id":"p3775"},{"id":"p3773"},{"id":"p3774"}],"margin":0,"sizing_mode":"stretch_width","align":"start"}},{"type":"object","name":"Figure","id":"p3784","attributes":{"width":700,"height":300,"margin":[5,10],"sizing_mode":"fixed","align":"start","x_range":{"type":"object","name":"Range1d","id":"p3777","attributes":{"tags":[[["k","k",null]],[]],"start":1.0,"end":10.0,"reset_start":1.0,"reset_end":10.0}},"y_range":{"type":"object","name":"Range1d","id":"p3778","attributes":{"tags":[[["inertia","inertia",null]],{"type":"map","entries":[["invert_yaxis",false],["autorange",false]]}],"start":-16.15241218548626,"end":298.3193461763283,"reset_start":-16.15241218548626,"reset_end":298.3193461763283}},"x_scale":{"type":"object","name":"LinearScale","id":"p3794"},"y_scale":{"type":"object","name":"LinearScale","id":"p3795"},"title":{"type":"object","name":"Title","id":"p3787","attributes":{"text":"Elbow Curve","text_color":"black","text_font_size":"12pt"}},"renderers":[{"type":"object","name":"GlyphRenderer","id":"p3820","attributes":{"data_source":{"type":"object","name":"ColumnDataSource","id":"p3811","attributes":{"selected":{"type":"object","name":"Selection","id":"p3812","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"p3813"},"data":{"type":"map","entries":[["k",{"type":"ndarray","array":{"type":"bytes","data":"AQAAAAIAAAADAAAABAAAAAUAAAAGAAAABwAAAAgAAAAJAAAACgAAAA=="},"shape":[10],"dtype":"int32","order":"little"}],["inertia",{"type":"ndarray","array":{"type":"bytes","data":"HOIxWdABcUCbsYa8YLFmQIwHLxP96lhAZkpm+1udR0DoqLMVBwlCQKSOjXW6UTpAD5lBlbj/M0C67ncrjJwwQGaPv/jPKCpAJkACOW0bJEA="},"shape":[10],"dtype":"float64","order":"little"}]]}}},"view":{"type":"object","name":"CDSView","id":"p3821","attributes":{"filter":{"type":"object","name":"AllIndices","id":"p3822"}}},"glyph":{"type":"object","name":"Line","id":"p3817","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"k"},"y":{"type":"field","field":"inertia"},"line_color":"#30a2da","line_width":2}},"selection_glyph":{"type":"object","name":"Line","id":"p3823","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"k"},"y":{"type":"field","field":"inertia"},"line_color":"#30a2da","line_width":2}},"nonselection_glyph":{"type":"object","name":"Line","id":"p3818","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"k"},"y":{"type":"field","field":"inertia"},"line_color":"#30a2da","line_alpha":0.1,"line_width":2}},"muted_glyph":{"type":"object","name":"Line","id":"p3819","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"k"},"y":{"type":"field","field":"inertia"},"line_color":"#30a2da","line_alpha":0.2,"line_width":2}}}}],"toolbar":{"type":"object","name":"Toolbar","id":"p3793","attributes":{"tools":[{"type":"object","name":"WheelZoomTool","id":"p3782","attributes":{"tags":["hv_created"],"zoom_together":"none"}},{"type":"object","name":"HoverTool","id":"p3783","attributes":{"tags":["hv_created"],"renderers":[{"id":"p3820"}],"tooltips":[["k","@{k}"],["inertia","@{inertia}"]]}},{"type":"object","name":"SaveTool","id":"p3806"},{"type":"object","name":"PanTool","id":"p3807"},{"type":"object","name":"BoxZoomTool","id":"p3808","attributes":{"overlay":{"type":"object","name":"BoxAnnotation","id":"p3809","attributes":{"syncable":false,"level":"overlay","visible":false,"left_units":"canvas","right_units":"canvas","bottom_units":"canvas","top_units":"canvas","line_color":"black","line_alpha":1.0,"line_width":2,"line_dash":[4,4],"fill_color":"lightgrey","fill_alpha":0.5}}}},{"type":"object","name":"ResetTool","id":"p3810"}],"active_drag":{"id":"p3807"},"active_scroll":{"id":"p3782"}}},"left":[{"type":"object","name":"LinearAxis","id":"p3801","attributes":{"ticker":{"type":"object","name":"BasicTicker","id":"p3802","attributes":{"mantissas":[1,2,5]}},"formatter":{"type":"object","name":"BasicTickFormatter","id":"p3803"},"axis_label":"inertia","major_label_policy":{"type":"object","name":"AllLabels","id":"p3804"}}}],"below":[{"type":"object","name":"LinearAxis","id":"p3796","attributes":{"ticker":{"type":"object","name":"FixedTicker","id":"p3824","attributes":{"ticks":[1,2,3,4,5,6,7,8,9,10],"minor_ticks":[]}},"formatter":{"type":"object","name":"BasicTickFormatter","id":"p3798"},"axis_label":"k","major_label_policy":{"type":"object","name":"AllLabels","id":"p3799"}}}],"center":[{"type":"object","name":"Grid","id":"p3800","attributes":{"axis":{"id":"p3796"},"grid_line_color":null}},{"type":"object","name":"Grid","id":"p3805","attributes":{"dimension":1,"axis":{"id":"p3801"},"grid_line_color":null}}],"min_border_top":10,"min_border_bottom":10,"min_border_left":10,"min_border_right":10,"output_backend":"webgl"}},{"type":"object","name":"Spacer","id":"p3826","attributes":{"name":"HSpacer07474","stylesheets":["\n:host(.pn-loading.pn-arc):before, .pn-loading.pn-arc:before {\n  background-image: url(\"data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHN0eWxlPSJtYXJnaW46IGF1dG87IGJhY2tncm91bmQ6IG5vbmU7IGRpc3BsYXk6IGJsb2NrOyBzaGFwZS1yZW5kZXJpbmc6IGF1dG87IiB2aWV3Qm94PSIwIDAgMTAwIDEwMCIgcHJlc2VydmVBc3BlY3RSYXRpbz0ieE1pZFlNaWQiPiAgPGNpcmNsZSBjeD0iNTAiIGN5PSI1MCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjYzNjM2MzIiBzdHJva2Utd2lkdGg9IjEwIiByPSIzNSIgc3Ryb2tlLWRhc2hhcnJheT0iMTY0LjkzMzYxNDMxMzQ2NDE1IDU2Ljk3Nzg3MTQzNzgyMTM4Ij4gICAgPGFuaW1hdGVUcmFuc2Zvcm0gYXR0cmlidXRlTmFtZT0idHJhbnNmb3JtIiB0eXBlPSJyb3RhdGUiIHJlcGVhdENvdW50PSJpbmRlZmluaXRlIiBkdXI9IjFzIiB2YWx1ZXM9IjAgNTAgNTA7MzYwIDUwIDUwIiBrZXlUaW1lcz0iMDsxIj48L2FuaW1hdGVUcmFuc2Zvcm0+ICA8L2NpcmNsZT48L3N2Zz4=\");\n  background-size: auto calc(min(50%, 400px));\n}",{"id":"p3775"},{"id":"p3773"},{"id":"p3774"}],"margin":0,"sizing_mode":"stretch_width","align":"start"}}]}}],"defs":[{"type":"model","name":"ReactiveHTML1"},{"type":"model","name":"FlexBox1","properties":[{"name":"align_content","kind":"Any","default":"flex-start"},{"name":"align_items","kind":"Any","default":"flex-start"},{"name":"flex_direction","kind":"Any","default":"row"},{"name":"flex_wrap","kind":"Any","default":"wrap"},{"name":"justify_content","kind":"Any","default":"flex-start"}]},{"type":"model","name":"FloatPanel1","properties":[{"name":"config","kind":"Any","default":{"type":"map"}},{"name":"contained","kind":"Any","default":true},{"name":"position","kind":"Any","default":"right-top"},{"name":"offsetx","kind":"Any","default":null},{"name":"offsety","kind":"Any","default":null},{"name":"theme","kind":"Any","default":"primary"},{"name":"status","kind":"Any","default":"normalized"}]},{"type":"model","name":"GridStack1","properties":[{"name":"mode","kind":"Any","default":"warn"},{"name":"ncols","kind":"Any","default":null},{"name":"nrows","kind":"Any","default":null},{"name":"allow_resize","kind":"Any","default":true},{"name":"allow_drag","kind":"Any","default":true},{"name":"state","kind":"Any","default":[]}]},{"type":"model","name":"drag1","properties":[{"name":"slider_width","kind":"Any","default":5},{"name":"slider_color","kind":"Any","default":"black"},{"name":"value","kind":"Any","default":50}]},{"type":"model","name":"click1","properties":[{"name":"terminal_output","kind":"Any","default":""},{"name":"debug_name","kind":"Any","default":""},{"name":"clears","kind":"Any","default":0}]},{"type":"model","name":"FastWrapper1","properties":[{"name":"object","kind":"Any","default":null},{"name":"style","kind":"Any","default":null}]},{"type":"model","name":"NotificationAreaBase1","properties":[{"name":"js_events","kind":"Any","default":{"type":"map"}},{"name":"position","kind":"Any","default":"bottom-right"},{"name":"_clear","kind":"Any","default":0}]},{"type":"model","name":"NotificationArea1","properties":[{"name":"js_events","kind":"Any","default":{"type":"map"}},{"name":"notifications","kind":"Any","default":[]},{"name":"position","kind":"Any","default":"bottom-right"},{"name":"_clear","kind":"Any","default":0},{"name":"types","kind":"Any","default":[{"type":"map","entries":[["type","warning"],["background","#ffc107"],["icon",{"type":"map","entries":[["className","fas fa-exclamation-triangle"],["tagName","i"],["color","white"]]}]]},{"type":"map","entries":[["type","info"],["background","#007bff"],["icon",{"type":"map","entries":[["className","fas fa-info-circle"],["tagName","i"],["color","white"]]}]]}]}]},{"type":"model","name":"Notification","properties":[{"name":"background","kind":"Any","default":null},{"name":"duration","kind":"Any","default":3000},{"name":"icon","kind":"Any","default":null},{"name":"message","kind":"Any","default":""},{"name":"notification_type","kind":"Any","default":null},{"name":"_destroyed","kind":"Any","default":false}]},{"type":"model","name":"TemplateActions1","properties":[{"name":"open_modal","kind":"Any","default":0},{"name":"close_modal","kind":"Any","default":0}]},{"type":"model","name":"BootstrapTemplateActions1","properties":[{"name":"open_modal","kind":"Any","default":0},{"name":"close_modal","kind":"Any","default":0}]},{"type":"model","name":"MaterialTemplateActions1","properties":[{"name":"open_modal","kind":"Any","default":0},{"name":"close_modal","kind":"Any","default":0}]}]}};
  var render_items = [{"docid":"a764cdd6-cda6-472f-b960-edfd2205b32a","roots":{"p3772":"f57cf9a0-4d67-4f57-8dc9-b1d56effefad"},"root_ids":["p3772"]}];
  var docs = Object.values(docs_json)
  if (!docs) {
    return
  }
  const py_version = docs[0].version.replace('rc', '-rc.').replace('.dev', '-dev.')
  const is_dev = py_version.indexOf("+") !== -1 || py_version.indexOf("-") !== -1
  function embed_document(root) {
    var Bokeh = get_bokeh(root)
    Bokeh.embed.embed_items_notebook(docs_json, render_items);
    for (const render_item of render_items) {
      for (const root_id of render_item.root_ids) {
	const id_el = document.getElementById(root_id)
	if (id_el.children.length && (id_el.children[0].className === 'bk-root')) {
	  const root_el = id_el.children[0]
	  root_el.id = root_el.id + '-rendered'
	}
      }
    }
  }
  function get_bokeh(root) {
    if (root.Bokeh === undefined) {
      return null
    } else if (root.Bokeh.version !== py_version && !is_dev) {
      if (root.Bokeh.versions === undefined || !root.Bokeh.versions.has(py_version)) {
	return null
      }
      return root.Bokeh.versions.get(py_version);
    } else if (root.Bokeh.version === py_version) {
      return root.Bokeh
    }
    return null
  }
  function is_loaded(root) {
    var Bokeh = get_bokeh(root)
    return (Bokeh != null && Bokeh.Panel !== undefined)
  }
  if (is_loaded(root)) {
    embed_document(root);
  } else {
    var attempts = 0;
    var timer = setInterval(function(root) {
      if (is_loaded(root)) {
        clearInterval(timer);
        embed_document(root);
      } else if (document.readyState == "complete") {
        attempts++;
        if (attempts > 200) {
          clearInterval(timer);
	  var Bokeh = get_bokeh(root)
	  if (Bokeh == null || Bokeh.Panel == null) {
            console.warn("Panel: ERROR: Unable to run Panel code because Bokeh or Panel library is missing");
	  } else {
	    console.warn("Panel: WARNING: Attempting to render but not all required libraries could be resolved.")
	    embed_document(root)
	  }
        }
      }
    }, 25, root)
  }
})(window);</script>



#### Answer the following questions: What is the best value for k when using the PCA data? Does it differ from the best k value found using the original data?
* **Question:** What is the best value for `k` when using the PCA data?

  * **Answer:** 4


* **Question:** Does it differ from the best k value found using the original data?

  * **Answer:** # NO

---

### Cluster Cryptocurrencies with K-means Using the PCA Data

In this section, you will use the PCA data and the K-Means algorithm with the best value for `k` found in the previous section to cluster the cryptocurrencies according to the principal components.

1. Initialize the K-Means model with four clusters using the best value for `k`. 

2. Fit the K-Means model using the PCA data.

3. Predict the clusters to group the cryptocurrencies using the PCA data. View the resulting array of cluster values.

4. Add a new column to the DataFrame with the PCA data to store the predicted clusters.

5. Create a scatter plot using hvPlot by setting `x="PC1"` and `y="PC2"`. Color the graph points with the labels found using K-Means and add the crypto name in the `hover_cols` parameter to identify the cryptocurrency represented by each data point.


```python
# Initialize the K-Means model using the best value for k
model = KMeans(n_clusters = 4)
```


```python
# Fit the K-Means model using the PCA data
model.fit(df_scaled_market_pca)
```

    C:\Users\HPs\anaconda3\Lib\site-packages\sklearn\cluster\_kmeans.py:1412: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      super()._check_params_vs_input(X, default_n_init=10)
    C:\Users\HPs\anaconda3\Lib\site-packages\sklearn\cluster\_kmeans.py:1436: UserWarning: KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS=1.
      warnings.warn(
    




<style>#sk-container-id-6 {color: black;}#sk-container-id-6 pre{padding: 0;}#sk-container-id-6 div.sk-toggleable {background-color: white;}#sk-container-id-6 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-6 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-6 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-6 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-6 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-6 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-6 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-6 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-6 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-6 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-6 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-6 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-6 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-6 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-6 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-6 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-6 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-6 div.sk-item {position: relative;z-index: 1;}#sk-container-id-6 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-6 div.sk-item::before, #sk-container-id-6 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-6 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-6 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-6 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-6 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-6 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-6 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-6 div.sk-label-container {text-align: center;}#sk-container-id-6 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-6 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-6" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>KMeans(n_clusters=4)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-6" type="checkbox" checked><label for="sk-estimator-id-6" class="sk-toggleable__label sk-toggleable__label-arrow">KMeans</label><div class="sk-toggleable__content"><pre>KMeans(n_clusters=4)</pre></div></div></div></div></div>




```python
# Predict the clusters to group the cryptocurrencies using the PCA data
data_predict_2 = model.predict(df_scaled_market_pca)

# View the resulting array of cluster values.
data_predict_2[:5]
```




    array([0, 0, 1, 1, 0])




```python
# Create a copy of the DataFrame with the PCA data
df_scaled_market_pca
# Add a new column to the DataFrame with the predicted clusters
df_scaled_market_pca["predicted clusters"] = data_predict_2

# Display sample data
df_scaled_market_pca.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price_change_percentage_24h</th>
      <th>price_change_percentage_7d</th>
      <th>price_change_percentage_14d</th>
      <th>predicted clusters</th>
    </tr>
    <tr>
      <th>coin_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>bitcoin</th>
      <td>0.102716</td>
      <td>-1.092940</td>
      <td>0.535184</td>
      <td>0</td>
    </tr>
    <tr>
      <th>ethereum</th>
      <td>0.076712</td>
      <td>-0.704779</td>
      <td>1.036855</td>
      <td>0</td>
    </tr>
    <tr>
      <th>tether</th>
      <td>-0.619999</td>
      <td>-0.019769</td>
      <td>-0.654034</td>
      <td>1</td>
    </tr>
    <tr>
      <th>ripple</th>
      <td>-0.664725</td>
      <td>0.002174</td>
      <td>-0.485221</td>
      <td>1</td>
    </tr>
    <tr>
      <th>bitcoin-cash</th>
      <td>0.288739</td>
      <td>-2.437009</td>
      <td>1.799157</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Create a scatter plot using hvPlot by setting 
# `x="PC1"` and `y="PC2"`. 
# Color the graph points with the labels found using K-Means and 
# add the crypto name in the `hover_cols` parameter to identify 
# the cryptocurrency represented by each data point.
df_scaled_market_pca.hvplot.scatter(
    x = "price_change_percentage_24h",
    y = "price_change_percentage_7d",
    by = "predicted clusters",
    hover_cols = "coin_id",
    title = "Segment Clusters",
    xlabel = "Price Change Percentage 24 Hours",
    ylabel = "Price Change Percentage 7 Days"
)
```






<div id='p4133'>
  <div id="c955ecd8-279c-4494-ba1f-6aded780f706" data-root-id="p4133" style="display: contents;"></div>
</div>
<script type="application/javascript">(function(root) {
  var docs_json = {"9bd1b30e-04f5-4cd4-a125-f1de86feb600":{"version":"3.2.1","title":"Bokeh Application","roots":[{"type":"object","name":"Row","id":"p4133","attributes":{"name":"Row08373","tags":["embedded"],"stylesheets":["\n:host(.pn-loading.pn-arc):before, .pn-loading.pn-arc:before {\n  background-image: url(\"data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHN0eWxlPSJtYXJnaW46IGF1dG87IGJhY2tncm91bmQ6IG5vbmU7IGRpc3BsYXk6IGJsb2NrOyBzaGFwZS1yZW5kZXJpbmc6IGF1dG87IiB2aWV3Qm94PSIwIDAgMTAwIDEwMCIgcHJlc2VydmVBc3BlY3RSYXRpbz0ieE1pZFlNaWQiPiAgPGNpcmNsZSBjeD0iNTAiIGN5PSI1MCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjYzNjM2MzIiBzdHJva2Utd2lkdGg9IjEwIiByPSIzNSIgc3Ryb2tlLWRhc2hhcnJheT0iMTY0LjkzMzYxNDMxMzQ2NDE1IDU2Ljk3Nzg3MTQzNzgyMTM4Ij4gICAgPGFuaW1hdGVUcmFuc2Zvcm0gYXR0cmlidXRlTmFtZT0idHJhbnNmb3JtIiB0eXBlPSJyb3RhdGUiIHJlcGVhdENvdW50PSJpbmRlZmluaXRlIiBkdXI9IjFzIiB2YWx1ZXM9IjAgNTAgNTA7MzYwIDUwIDUwIiBrZXlUaW1lcz0iMDsxIj48L2FuaW1hdGVUcmFuc2Zvcm0+ICA8L2NpcmNsZT48L3N2Zz4=\");\n  background-size: auto calc(min(50%, 400px));\n}",{"type":"object","name":"ImportedStyleSheet","id":"p4136","attributes":{"url":"https://cdn.holoviz.org/panel/1.2.1/dist/css/loading.css"}},{"type":"object","name":"ImportedStyleSheet","id":"p4241","attributes":{"url":"https://cdn.holoviz.org/panel/1.2.1/dist/css/listpanel.css"}},{"type":"object","name":"ImportedStyleSheet","id":"p4134","attributes":{"url":"https://cdn.holoviz.org/panel/1.2.1/dist/bundled/theme/default.css"}},{"type":"object","name":"ImportedStyleSheet","id":"p4135","attributes":{"url":"https://cdn.holoviz.org/panel/1.2.1/dist/bundled/theme/native.css"}}],"min_width":700,"margin":0,"sizing_mode":"stretch_width","align":"start","children":[{"type":"object","name":"Spacer","id":"p4137","attributes":{"name":"HSpacer08384","stylesheets":["\n:host(.pn-loading.pn-arc):before, .pn-loading.pn-arc:before {\n  background-image: url(\"data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHN0eWxlPSJtYXJnaW46IGF1dG87IGJhY2tncm91bmQ6IG5vbmU7IGRpc3BsYXk6IGJsb2NrOyBzaGFwZS1yZW5kZXJpbmc6IGF1dG87IiB2aWV3Qm94PSIwIDAgMTAwIDEwMCIgcHJlc2VydmVBc3BlY3RSYXRpbz0ieE1pZFlNaWQiPiAgPGNpcmNsZSBjeD0iNTAiIGN5PSI1MCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjYzNjM2MzIiBzdHJva2Utd2lkdGg9IjEwIiByPSIzNSIgc3Ryb2tlLWRhc2hhcnJheT0iMTY0LjkzMzYxNDMxMzQ2NDE1IDU2Ljk3Nzg3MTQzNzgyMTM4Ij4gICAgPGFuaW1hdGVUcmFuc2Zvcm0gYXR0cmlidXRlTmFtZT0idHJhbnNmb3JtIiB0eXBlPSJyb3RhdGUiIHJlcGVhdENvdW50PSJpbmRlZmluaXRlIiBkdXI9IjFzIiB2YWx1ZXM9IjAgNTAgNTA7MzYwIDUwIDUwIiBrZXlUaW1lcz0iMDsxIj48L2FuaW1hdGVUcmFuc2Zvcm0+ICA8L2NpcmNsZT48L3N2Zz4=\");\n  background-size: auto calc(min(50%, 400px));\n}",{"id":"p4136"},{"id":"p4134"},{"id":"p4135"}],"margin":0,"sizing_mode":"stretch_width","align":"start"}},{"type":"object","name":"Figure","id":"p4154","attributes":{"width":700,"height":300,"margin":[5,10],"sizing_mode":"fixed","align":"start","x_range":{"type":"object","name":"Range1d","id":"p4138","attributes":{"tags":[[["price_change_percentage_24h","price_change_percentage_24h",null]],[]],"start":-2.080090654048831,"end":8.157012809779975,"reset_start":-2.080090654048831,"reset_end":8.157012809779975}},"y_range":{"type":"object","name":"Range1d","id":"p4139","attributes":{"tags":[[["price_change_percentage_7d","price_change_percentage_7d",null]],{"type":"map","entries":[["invert_yaxis",false],["autorange",false]]}],"start":-4.582209243538451,"end":8.340274168490136,"reset_start":-4.582209243538451,"reset_end":8.340274168490136}},"x_scale":{"type":"object","name":"LinearScale","id":"p4164"},"y_scale":{"type":"object","name":"LinearScale","id":"p4165"},"title":{"type":"object","name":"Title","id":"p4157","attributes":{"text":"Segment Clusters","text_color":"black","text_font_size":"12pt"}},"renderers":[{"type":"object","name":"GlyphRenderer","id":"p4190","attributes":{"name":"0","data_source":{"type":"object","name":"ColumnDataSource","id":"p4181","attributes":{"selected":{"type":"object","name":"Selection","id":"p4182","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"p4183"},"data":{"type":"map","entries":[["price_change_percentage_24h",{"type":"ndarray","array":{"type":"bytes","data":"PYEWfZ9Luj/igSoNYqOzP7jRiMaxetI/Fgrh4gxS3D+NasEmYpDOPysFYCd8LtU/y0hran0Gvr8jAPanmaH6Pw3tdw6zK+S/AKU1OyJR1z89qaf9yFq3P+HxjfQqScA/nqxLEE6Io78="},"shape":[13],"dtype":"float64","order":"little"}],["price_change_percentage_7d",{"type":"ndarray","array":{"type":"bytes","data":"urMJzq588b9h/uLSjI3mvwvgJ23+fgPAqC/ykVJc+L802JJu0Zzpv2B/Eu+ciuy/+Ck8Uu2u6L+evjykvVMEwGWKJpxGzbG/8KBlSoRY67+xyOCyUEzxv31mpzwdCue/LwIt7iXp2b8="},"shape":[13],"dtype":"float64","order":"little"}],["coin_id",["bitcoin","ethereum","bitcoin-cash","binancecoin","chainlink","cardano","litecoin","monero","tezos","cosmos","wrapped-bitcoin","zcash","maker"]],["predicted_clusters",[0,0,0,0,0,0,0,0,0,0,0,0,0]]]}}},"view":{"type":"object","name":"CDSView","id":"p4191","attributes":{"filter":{"type":"object","name":"AllIndices","id":"p4192"}}},"glyph":{"type":"object","name":"Scatter","id":"p4187","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"price_change_percentage_24h"},"y":{"type":"field","field":"price_change_percentage_7d"},"size":{"type":"value","value":5.477225575051661},"line_color":{"type":"value","value":"#30a2da"},"fill_color":{"type":"value","value":"#30a2da"},"hatch_color":{"type":"value","value":"#30a2da"}}},"selection_glyph":{"type":"object","name":"Scatter","id":"p4195","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"price_change_percentage_24h"},"y":{"type":"field","field":"price_change_percentage_7d"},"size":{"type":"value","value":5.477225575051661},"angle":{"type":"value","value":0.0},"line_color":{"type":"value","value":"#30a2da"},"line_alpha":{"type":"value","value":1.0},"line_width":{"type":"value","value":1},"line_join":{"type":"value","value":"bevel"},"line_cap":{"type":"value","value":"butt"},"line_dash":{"type":"value","value":[]},"line_dash_offset":{"type":"value","value":0},"fill_color":{"type":"value","value":"#30a2da"},"fill_alpha":{"type":"value","value":1.0},"hatch_color":{"type":"value","value":"#30a2da"},"hatch_alpha":{"type":"value","value":1.0},"hatch_scale":{"type":"value","value":12.0},"hatch_pattern":{"type":"value","value":null},"hatch_weight":{"type":"value","value":1.0},"marker":{"type":"value","value":"circle"}}},"nonselection_glyph":{"type":"object","name":"Scatter","id":"p4188","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"price_change_percentage_24h"},"y":{"type":"field","field":"price_change_percentage_7d"},"size":{"type":"value","value":5.477225575051661},"line_color":{"type":"value","value":"#30a2da"},"line_alpha":{"type":"value","value":0.1},"fill_color":{"type":"value","value":"#30a2da"},"fill_alpha":{"type":"value","value":0.1},"hatch_color":{"type":"value","value":"#30a2da"},"hatch_alpha":{"type":"value","value":0.1}}},"muted_glyph":{"type":"object","name":"Scatter","id":"p4189","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"price_change_percentage_24h"},"y":{"type":"field","field":"price_change_percentage_7d"},"size":{"type":"value","value":5.477225575051661},"line_color":{"type":"value","value":"#30a2da"},"line_alpha":{"type":"value","value":0.2},"fill_color":{"type":"value","value":"#30a2da"},"fill_alpha":{"type":"value","value":0.2},"hatch_color":{"type":"value","value":"#30a2da"},"hatch_alpha":{"type":"value","value":0.2}}}}},{"type":"object","name":"GlyphRenderer","id":"p4205","attributes":{"name":"1","data_source":{"type":"object","name":"ColumnDataSource","id":"p4196","attributes":{"selected":{"type":"object","name":"Selection","id":"p4197","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"p4198"},"data":{"type":"map","entries":[["price_change_percentage_24h",{"type":"ndarray","array":{"type":"bytes","data":"nGyQQgfX478pSYPDbEXlv3owJkhA6uu/bj7D2hwx8b/nDqtxxSHkvz3/ePHv4e6/QQav7I2X4L+5Rfvqeu7nv8loLIxVUey/O0g9rACf5b/k3WkGGzjuv0IkCmugouu/SuNrVqAx47+2MS8bgh/hPxHWGIfcDOS/QhAmCKHP778GuhAvPB75v0lozcaR0P4/j8PZqVsP678vhJO800fpvxG0ohWcgdy/Sd6OEdarsD8NvGGS39D6v2YnnZwYd+O/Y9l4MCQC5L8ZUo+f6ujgvw=="},"shape":[26],"dtype":"float64","order":"little"}],["price_change_percentage_7d",{"type":"ndarray","array":{"type":"bytes","data":"qKV1G1k+lL+4jSIpJ89hPxu8vi3S4cW/Mpn6aWEA8j9z0tFdaTSQvz1gIz6zdbo/hKMFK4/zsL/rM3NTlf3yP7WyW+gMt9o/mcMX3mETtb8QQNGrt/PmP13UOEjIqts/f96jznRamD+BPCC+xNW3v+uD0FfGCaa/UMsfnHrWtT9DZsFUngj3PwAWcGTTr/c/wXduBhVMwD/x0u4oRh6qP5yTdUvdQgNAvC8ePl3xzb+rscH76IL4P4EqY26iQdK/6lea4uDzrr87fb92BbmyPw=="},"shape":[26],"dtype":"float64","order":"little"}],["coin_id",["tether","ripple","bitcoin-cash-sv","crypto-com-chain","usd-coin","eos","tron","okb","stellar","cdai","neo","leo-token","huobi-token","nem","binance-usd","iota","vechain","theta-token","dash","ethereum-classic","havven","omisego","ontology","ftx-token","true-usd","digibyte"]],["predicted_clusters",[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]]]}}},"view":{"type":"object","name":"CDSView","id":"p4206","attributes":{"filter":{"type":"object","name":"AllIndices","id":"p4207"}}},"glyph":{"type":"object","name":"Scatter","id":"p4202","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"price_change_percentage_24h"},"y":{"type":"field","field":"price_change_percentage_7d"},"size":{"type":"value","value":5.477225575051661},"line_color":{"type":"value","value":"#fc4f30"},"fill_color":{"type":"value","value":"#fc4f30"},"hatch_color":{"type":"value","value":"#fc4f30"}}},"selection_glyph":{"type":"object","name":"Scatter","id":"p4209","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"price_change_percentage_24h"},"y":{"type":"field","field":"price_change_percentage_7d"},"size":{"type":"value","value":5.477225575051661},"angle":{"type":"value","value":0.0},"line_color":{"type":"value","value":"#fc4f30"},"line_alpha":{"type":"value","value":1.0},"line_width":{"type":"value","value":1},"line_join":{"type":"value","value":"bevel"},"line_cap":{"type":"value","value":"butt"},"line_dash":{"type":"value","value":[]},"line_dash_offset":{"type":"value","value":0},"fill_color":{"type":"value","value":"#fc4f30"},"fill_alpha":{"type":"value","value":1.0},"hatch_color":{"type":"value","value":"#fc4f30"},"hatch_alpha":{"type":"value","value":1.0},"hatch_scale":{"type":"value","value":12.0},"hatch_pattern":{"type":"value","value":null},"hatch_weight":{"type":"value","value":1.0},"marker":{"type":"value","value":"circle"}}},"nonselection_glyph":{"type":"object","name":"Scatter","id":"p4203","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"price_change_percentage_24h"},"y":{"type":"field","field":"price_change_percentage_7d"},"size":{"type":"value","value":5.477225575051661},"line_color":{"type":"value","value":"#fc4f30"},"line_alpha":{"type":"value","value":0.1},"fill_color":{"type":"value","value":"#fc4f30"},"fill_alpha":{"type":"value","value":0.1},"hatch_color":{"type":"value","value":"#fc4f30"},"hatch_alpha":{"type":"value","value":0.1}}},"muted_glyph":{"type":"object","name":"Scatter","id":"p4204","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"price_change_percentage_24h"},"y":{"type":"field","field":"price_change_percentage_7d"},"size":{"type":"value","value":5.477225575051661},"line_color":{"type":"value","value":"#fc4f30"},"line_alpha":{"type":"value","value":0.2},"fill_color":{"type":"value","value":"#fc4f30"},"fill_alpha":{"type":"value","value":0.2},"hatch_color":{"type":"value","value":"#fc4f30"},"hatch_alpha":{"type":"value","value":0.2}}}}},{"type":"object","name":"GlyphRenderer","id":"p4219","attributes":{"name":"2","data_source":{"type":"object","name":"ColumnDataSource","id":"p4210","attributes":{"selected":{"type":"object","name":"Selection","id":"p4211","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"p4212"},"data":{"type":"map","entries":[["price_change_percentage_24h",{"type":"ndarray","array":{"type":"bytes","data":"vvRIi0cyFkA="},"shape":[1],"dtype":"float64","order":"little"}],["price_change_percentage_7d",{"type":"ndarray","array":{"type":"bytes","data":"jxm+37gNHUA="},"shape":[1],"dtype":"float64","order":"little"}],["coin_id",["ethlend"]],["predicted_clusters",[2]]]}}},"view":{"type":"object","name":"CDSView","id":"p4220","attributes":{"filter":{"type":"object","name":"AllIndices","id":"p4221"}}},"glyph":{"type":"object","name":"Scatter","id":"p4216","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"price_change_percentage_24h"},"y":{"type":"field","field":"price_change_percentage_7d"},"size":{"type":"value","value":5.477225575051661},"line_color":{"type":"value","value":"#e5ae38"},"fill_color":{"type":"value","value":"#e5ae38"},"hatch_color":{"type":"value","value":"#e5ae38"}}},"selection_glyph":{"type":"object","name":"Scatter","id":"p4223","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"price_change_percentage_24h"},"y":{"type":"field","field":"price_change_percentage_7d"},"size":{"type":"value","value":5.477225575051661},"angle":{"type":"value","value":0.0},"line_color":{"type":"value","value":"#e5ae38"},"line_alpha":{"type":"value","value":1.0},"line_width":{"type":"value","value":1},"line_join":{"type":"value","value":"bevel"},"line_cap":{"type":"value","value":"butt"},"line_dash":{"type":"value","value":[]},"line_dash_offset":{"type":"value","value":0},"fill_color":{"type":"value","value":"#e5ae38"},"fill_alpha":{"type":"value","value":1.0},"hatch_color":{"type":"value","value":"#e5ae38"},"hatch_alpha":{"type":"value","value":1.0},"hatch_scale":{"type":"value","value":12.0},"hatch_pattern":{"type":"value","value":null},"hatch_weight":{"type":"value","value":1.0},"marker":{"type":"value","value":"circle"}}},"nonselection_glyph":{"type":"object","name":"Scatter","id":"p4217","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"price_change_percentage_24h"},"y":{"type":"field","field":"price_change_percentage_7d"},"size":{"type":"value","value":5.477225575051661},"line_color":{"type":"value","value":"#e5ae38"},"line_alpha":{"type":"value","value":0.1},"fill_color":{"type":"value","value":"#e5ae38"},"fill_alpha":{"type":"value","value":0.1},"hatch_color":{"type":"value","value":"#e5ae38"},"hatch_alpha":{"type":"value","value":0.1}}},"muted_glyph":{"type":"object","name":"Scatter","id":"p4218","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"price_change_percentage_24h"},"y":{"type":"field","field":"price_change_percentage_7d"},"size":{"type":"value","value":5.477225575051661},"line_color":{"type":"value","value":"#e5ae38"},"line_alpha":{"type":"value","value":0.2},"fill_color":{"type":"value","value":"#e5ae38"},"fill_alpha":{"type":"value","value":0.2},"hatch_color":{"type":"value","value":"#e5ae38"},"hatch_alpha":{"type":"value","value":0.2}}}}},{"type":"object","name":"GlyphRenderer","id":"p4233","attributes":{"name":"3","data_source":{"type":"object","name":"ColumnDataSource","id":"p4224","attributes":{"selected":{"type":"object","name":"Selection","id":"p4225","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"p4226"},"data":{"type":"map","entries":[["price_change_percentage_24h",{"type":"ndarray","array":{"type":"bytes","data":"+D4Vk/wCH0A="},"shape":[1],"dtype":"float64","order":"little"}],["price_change_percentage_7d",{"type":"ndarray","array":{"type":"bytes","data":"9uaUZ+0KDMA="},"shape":[1],"dtype":"float64","order":"little"}],["coin_id",["celsius-degree-token"]],["predicted_clusters",[3]]]}}},"view":{"type":"object","name":"CDSView","id":"p4234","attributes":{"filter":{"type":"object","name":"AllIndices","id":"p4235"}}},"glyph":{"type":"object","name":"Scatter","id":"p4230","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"price_change_percentage_24h"},"y":{"type":"field","field":"price_change_percentage_7d"},"size":{"type":"value","value":5.477225575051661},"line_color":{"type":"value","value":"#6d904f"},"fill_color":{"type":"value","value":"#6d904f"},"hatch_color":{"type":"value","value":"#6d904f"}}},"selection_glyph":{"type":"object","name":"Scatter","id":"p4237","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"price_change_percentage_24h"},"y":{"type":"field","field":"price_change_percentage_7d"},"size":{"type":"value","value":5.477225575051661},"angle":{"type":"value","value":0.0},"line_color":{"type":"value","value":"#6d904f"},"line_alpha":{"type":"value","value":1.0},"line_width":{"type":"value","value":1},"line_join":{"type":"value","value":"bevel"},"line_cap":{"type":"value","value":"butt"},"line_dash":{"type":"value","value":[]},"line_dash_offset":{"type":"value","value":0},"fill_color":{"type":"value","value":"#6d904f"},"fill_alpha":{"type":"value","value":1.0},"hatch_color":{"type":"value","value":"#6d904f"},"hatch_alpha":{"type":"value","value":1.0},"hatch_scale":{"type":"value","value":12.0},"hatch_pattern":{"type":"value","value":null},"hatch_weight":{"type":"value","value":1.0},"marker":{"type":"value","value":"circle"}}},"nonselection_glyph":{"type":"object","name":"Scatter","id":"p4231","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"price_change_percentage_24h"},"y":{"type":"field","field":"price_change_percentage_7d"},"size":{"type":"value","value":5.477225575051661},"line_color":{"type":"value","value":"#6d904f"},"line_alpha":{"type":"value","value":0.1},"fill_color":{"type":"value","value":"#6d904f"},"fill_alpha":{"type":"value","value":0.1},"hatch_color":{"type":"value","value":"#6d904f"},"hatch_alpha":{"type":"value","value":0.1}}},"muted_glyph":{"type":"object","name":"Scatter","id":"p4232","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"price_change_percentage_24h"},"y":{"type":"field","field":"price_change_percentage_7d"},"size":{"type":"value","value":5.477225575051661},"line_color":{"type":"value","value":"#6d904f"},"line_alpha":{"type":"value","value":0.2},"fill_color":{"type":"value","value":"#6d904f"},"fill_alpha":{"type":"value","value":0.2},"hatch_color":{"type":"value","value":"#6d904f"},"hatch_alpha":{"type":"value","value":0.2}}}}}],"toolbar":{"type":"object","name":"Toolbar","id":"p4163","attributes":{"tools":[{"type":"object","name":"WheelZoomTool","id":"p4143","attributes":{"tags":["hv_created"],"zoom_together":"none"}},{"type":"object","name":"HoverTool","id":"p4144","attributes":{"tags":["hv_created"],"renderers":[{"id":"p4190"},{"id":"p4205"},{"id":"p4219"},{"id":"p4233"}],"tooltips":[["predicted clusters","@{predicted_clusters}"],["price_change_percentage_24h","@{price_change_percentage_24h}"],["price_change_percentage_7d","@{price_change_percentage_7d}"],["coin_id","@{coin_id}"]]}},{"type":"object","name":"SaveTool","id":"p4176"},{"type":"object","name":"PanTool","id":"p4177"},{"type":"object","name":"BoxZoomTool","id":"p4178","attributes":{"overlay":{"type":"object","name":"BoxAnnotation","id":"p4179","attributes":{"syncable":false,"level":"overlay","visible":false,"left_units":"canvas","right_units":"canvas","bottom_units":"canvas","top_units":"canvas","line_color":"black","line_alpha":1.0,"line_width":2,"line_dash":[4,4],"fill_color":"lightgrey","fill_alpha":0.5}}}},{"type":"object","name":"ResetTool","id":"p4180"}],"active_drag":{"id":"p4177"},"active_scroll":{"id":"p4143"}}},"left":[{"type":"object","name":"LinearAxis","id":"p4171","attributes":{"ticker":{"type":"object","name":"BasicTicker","id":"p4172","attributes":{"mantissas":[1,2,5]}},"formatter":{"type":"object","name":"BasicTickFormatter","id":"p4173"},"axis_label":"Price Change Percentage 7 Days","major_label_policy":{"type":"object","name":"AllLabels","id":"p4174"}}}],"right":[{"type":"object","name":"Legend","id":"p4193","attributes":{"location":[0,0],"title":"predicted clusters","click_policy":"mute","items":[{"type":"object","name":"LegendItem","id":"p4194","attributes":{"label":{"type":"value","value":"0"},"renderers":[{"id":"p4190"}]}},{"type":"object","name":"LegendItem","id":"p4208","attributes":{"label":{"type":"value","value":"1"},"renderers":[{"id":"p4205"}]}},{"type":"object","name":"LegendItem","id":"p4222","attributes":{"label":{"type":"value","value":"2"},"renderers":[{"id":"p4219"}]}},{"type":"object","name":"LegendItem","id":"p4236","attributes":{"label":{"type":"value","value":"3"},"renderers":[{"id":"p4233"}]}}]}}],"below":[{"type":"object","name":"LinearAxis","id":"p4166","attributes":{"ticker":{"type":"object","name":"BasicTicker","id":"p4167","attributes":{"mantissas":[1,2,5]}},"formatter":{"type":"object","name":"BasicTickFormatter","id":"p4168"},"axis_label":"Price Change Percentage 24 Hours","major_label_policy":{"type":"object","name":"AllLabels","id":"p4169"}}}],"center":[{"type":"object","name":"Grid","id":"p4170","attributes":{"axis":{"id":"p4166"},"grid_line_color":null}},{"type":"object","name":"Grid","id":"p4175","attributes":{"dimension":1,"axis":{"id":"p4171"},"grid_line_color":null}}],"min_border_top":10,"min_border_bottom":10,"min_border_left":10,"min_border_right":10,"output_backend":"webgl"}},{"type":"object","name":"Spacer","id":"p4239","attributes":{"name":"HSpacer08387","stylesheets":["\n:host(.pn-loading.pn-arc):before, .pn-loading.pn-arc:before {\n  background-image: url(\"data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHN0eWxlPSJtYXJnaW46IGF1dG87IGJhY2tncm91bmQ6IG5vbmU7IGRpc3BsYXk6IGJsb2NrOyBzaGFwZS1yZW5kZXJpbmc6IGF1dG87IiB2aWV3Qm94PSIwIDAgMTAwIDEwMCIgcHJlc2VydmVBc3BlY3RSYXRpbz0ieE1pZFlNaWQiPiAgPGNpcmNsZSBjeD0iNTAiIGN5PSI1MCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjYzNjM2MzIiBzdHJva2Utd2lkdGg9IjEwIiByPSIzNSIgc3Ryb2tlLWRhc2hhcnJheT0iMTY0LjkzMzYxNDMxMzQ2NDE1IDU2Ljk3Nzg3MTQzNzgyMTM4Ij4gICAgPGFuaW1hdGVUcmFuc2Zvcm0gYXR0cmlidXRlTmFtZT0idHJhbnNmb3JtIiB0eXBlPSJyb3RhdGUiIHJlcGVhdENvdW50PSJpbmRlZmluaXRlIiBkdXI9IjFzIiB2YWx1ZXM9IjAgNTAgNTA7MzYwIDUwIDUwIiBrZXlUaW1lcz0iMDsxIj48L2FuaW1hdGVUcmFuc2Zvcm0+ICA8L2NpcmNsZT48L3N2Zz4=\");\n  background-size: auto calc(min(50%, 400px));\n}",{"id":"p4136"},{"id":"p4134"},{"id":"p4135"}],"margin":0,"sizing_mode":"stretch_width","align":"start"}}]}}],"defs":[{"type":"model","name":"ReactiveHTML1"},{"type":"model","name":"FlexBox1","properties":[{"name":"align_content","kind":"Any","default":"flex-start"},{"name":"align_items","kind":"Any","default":"flex-start"},{"name":"flex_direction","kind":"Any","default":"row"},{"name":"flex_wrap","kind":"Any","default":"wrap"},{"name":"justify_content","kind":"Any","default":"flex-start"}]},{"type":"model","name":"FloatPanel1","properties":[{"name":"config","kind":"Any","default":{"type":"map"}},{"name":"contained","kind":"Any","default":true},{"name":"position","kind":"Any","default":"right-top"},{"name":"offsetx","kind":"Any","default":null},{"name":"offsety","kind":"Any","default":null},{"name":"theme","kind":"Any","default":"primary"},{"name":"status","kind":"Any","default":"normalized"}]},{"type":"model","name":"GridStack1","properties":[{"name":"mode","kind":"Any","default":"warn"},{"name":"ncols","kind":"Any","default":null},{"name":"nrows","kind":"Any","default":null},{"name":"allow_resize","kind":"Any","default":true},{"name":"allow_drag","kind":"Any","default":true},{"name":"state","kind":"Any","default":[]}]},{"type":"model","name":"drag1","properties":[{"name":"slider_width","kind":"Any","default":5},{"name":"slider_color","kind":"Any","default":"black"},{"name":"value","kind":"Any","default":50}]},{"type":"model","name":"click1","properties":[{"name":"terminal_output","kind":"Any","default":""},{"name":"debug_name","kind":"Any","default":""},{"name":"clears","kind":"Any","default":0}]},{"type":"model","name":"FastWrapper1","properties":[{"name":"object","kind":"Any","default":null},{"name":"style","kind":"Any","default":null}]},{"type":"model","name":"NotificationAreaBase1","properties":[{"name":"js_events","kind":"Any","default":{"type":"map"}},{"name":"position","kind":"Any","default":"bottom-right"},{"name":"_clear","kind":"Any","default":0}]},{"type":"model","name":"NotificationArea1","properties":[{"name":"js_events","kind":"Any","default":{"type":"map"}},{"name":"notifications","kind":"Any","default":[]},{"name":"position","kind":"Any","default":"bottom-right"},{"name":"_clear","kind":"Any","default":0},{"name":"types","kind":"Any","default":[{"type":"map","entries":[["type","warning"],["background","#ffc107"],["icon",{"type":"map","entries":[["className","fas fa-exclamation-triangle"],["tagName","i"],["color","white"]]}]]},{"type":"map","entries":[["type","info"],["background","#007bff"],["icon",{"type":"map","entries":[["className","fas fa-info-circle"],["tagName","i"],["color","white"]]}]]}]}]},{"type":"model","name":"Notification","properties":[{"name":"background","kind":"Any","default":null},{"name":"duration","kind":"Any","default":3000},{"name":"icon","kind":"Any","default":null},{"name":"message","kind":"Any","default":""},{"name":"notification_type","kind":"Any","default":null},{"name":"_destroyed","kind":"Any","default":false}]},{"type":"model","name":"TemplateActions1","properties":[{"name":"open_modal","kind":"Any","default":0},{"name":"close_modal","kind":"Any","default":0}]},{"type":"model","name":"BootstrapTemplateActions1","properties":[{"name":"open_modal","kind":"Any","default":0},{"name":"close_modal","kind":"Any","default":0}]},{"type":"model","name":"MaterialTemplateActions1","properties":[{"name":"open_modal","kind":"Any","default":0},{"name":"close_modal","kind":"Any","default":0}]}]}};
  var render_items = [{"docid":"9bd1b30e-04f5-4cd4-a125-f1de86feb600","roots":{"p4133":"c955ecd8-279c-4494-ba1f-6aded780f706"},"root_ids":["p4133"]}];
  var docs = Object.values(docs_json)
  if (!docs) {
    return
  }
  const py_version = docs[0].version.replace('rc', '-rc.').replace('.dev', '-dev.')
  const is_dev = py_version.indexOf("+") !== -1 || py_version.indexOf("-") !== -1
  function embed_document(root) {
    var Bokeh = get_bokeh(root)
    Bokeh.embed.embed_items_notebook(docs_json, render_items);
    for (const render_item of render_items) {
      for (const root_id of render_item.root_ids) {
	const id_el = document.getElementById(root_id)
	if (id_el.children.length && (id_el.children[0].className === 'bk-root')) {
	  const root_el = id_el.children[0]
	  root_el.id = root_el.id + '-rendered'
	}
      }
    }
  }
  function get_bokeh(root) {
    if (root.Bokeh === undefined) {
      return null
    } else if (root.Bokeh.version !== py_version && !is_dev) {
      if (root.Bokeh.versions === undefined || !root.Bokeh.versions.has(py_version)) {
	return null
      }
      return root.Bokeh.versions.get(py_version);
    } else if (root.Bokeh.version === py_version) {
      return root.Bokeh
    }
    return null
  }
  function is_loaded(root) {
    var Bokeh = get_bokeh(root)
    return (Bokeh != null && Bokeh.Panel !== undefined)
  }
  if (is_loaded(root)) {
    embed_document(root);
  } else {
    var attempts = 0;
    var timer = setInterval(function(root) {
      if (is_loaded(root)) {
        clearInterval(timer);
        embed_document(root);
      } else if (document.readyState == "complete") {
        attempts++;
        if (attempts > 200) {
          clearInterval(timer);
	  var Bokeh = get_bokeh(root)
	  if (Bokeh == null || Bokeh.Panel == null) {
            console.warn("Panel: ERROR: Unable to run Panel code because Bokeh or Panel library is missing");
	  } else {
	    console.warn("Panel: WARNING: Attempting to render but not all required libraries could be resolved.")
	    embed_document(root)
	  }
        }
      }
    }, 25, root)
  }
})(window);</script>



### Visualize and Compare the Results

In this section, you will visually analyze the cluster analysis results by contrasting the outcome with and without using the optimization techniques.

1. Create a composite plot using hvPlot and the plus (`+`) operator to contrast the Elbow Curve that you created to find the best value for `k` with the original and the PCA data.

2. Create a composite plot using hvPlot and the plus (`+`) operator to contrast the cryptocurrencies clusters using the original and the PCA data.

3. Answer the following question: After visually analyzing the cluster analysis results, what is the impact of using fewer features to cluster the data using K-Means?

> **Rewind:** Back in Lesson 3 of Module 6, you learned how to create composite plots. You can look at that lesson to review how to make these plots; also, you can check [the hvPlot documentation](https://holoviz.org/tutorial/Composing_Plots.html).


```python
# Composite plot to contrast the Elbow curves
df_elbow.hvplot.line(x = "k", y = "inertia", 
                     title = "Elbow Curve", 
                     xticks = k
) + elbow_data_2.hvplot.line(x = "k", y = "inertia", 
                             title = "Elbow Curve", 
                             xticks = k,
)
```






<div id='p4995'>
  <div id="a2d8d0a7-fbf2-47b1-be38-6ec4c532f6be" data-root-id="p4995" style="display: contents;"></div>
</div>
<script type="application/javascript">(function(root) {
  var docs_json = {"3a2e5fee-2e04-40b9-b709-abc4174a3352":{"version":"3.2.1","title":"Bokeh Application","roots":[{"type":"object","name":"Row","id":"p4995","attributes":{"name":"Row10587","tags":["embedded"],"stylesheets":["\n:host(.pn-loading.pn-arc):before, .pn-loading.pn-arc:before {\n  background-image: url(\"data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHN0eWxlPSJtYXJnaW46IGF1dG87IGJhY2tncm91bmQ6IG5vbmU7IGRpc3BsYXk6IGJsb2NrOyBzaGFwZS1yZW5kZXJpbmc6IGF1dG87IiB2aWV3Qm94PSIwIDAgMTAwIDEwMCIgcHJlc2VydmVBc3BlY3RSYXRpbz0ieE1pZFlNaWQiPiAgPGNpcmNsZSBjeD0iNTAiIGN5PSI1MCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjYzNjM2MzIiBzdHJva2Utd2lkdGg9IjEwIiByPSIzNSIgc3Ryb2tlLWRhc2hhcnJheT0iMTY0LjkzMzYxNDMxMzQ2NDE1IDU2Ljk3Nzg3MTQzNzgyMTM4Ij4gICAgPGFuaW1hdGVUcmFuc2Zvcm0gYXR0cmlidXRlTmFtZT0idHJhbnNmb3JtIiB0eXBlPSJyb3RhdGUiIHJlcGVhdENvdW50PSJpbmRlZmluaXRlIiBkdXI9IjFzIiB2YWx1ZXM9IjAgNTAgNTA7MzYwIDUwIDUwIiBrZXlUaW1lcz0iMDsxIj48L2FuaW1hdGVUcmFuc2Zvcm0+ICA8L2NpcmNsZT48L3N2Zz4=\");\n  background-size: auto calc(min(50%, 400px));\n}",{"type":"object","name":"ImportedStyleSheet","id":"p4998","attributes":{"url":"https://cdn.holoviz.org/panel/1.2.1/dist/css/loading.css"}},{"type":"object","name":"ImportedStyleSheet","id":"p5113","attributes":{"url":"https://cdn.holoviz.org/panel/1.2.1/dist/css/listpanel.css"}},{"type":"object","name":"ImportedStyleSheet","id":"p4996","attributes":{"url":"https://cdn.holoviz.org/panel/1.2.1/dist/bundled/theme/default.css"}},{"type":"object","name":"ImportedStyleSheet","id":"p4997","attributes":{"url":"https://cdn.holoviz.org/panel/1.2.1/dist/bundled/theme/native.css"}}],"margin":0,"sizing_mode":"stretch_width","align":"start","children":[{"type":"object","name":"Spacer","id":"p4999","attributes":{"name":"HSpacer10597","stylesheets":["\n:host(.pn-loading.pn-arc):before, .pn-loading.pn-arc:before {\n  background-image: url(\"data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHN0eWxlPSJtYXJnaW46IGF1dG87IGJhY2tncm91bmQ6IG5vbmU7IGRpc3BsYXk6IGJsb2NrOyBzaGFwZS1yZW5kZXJpbmc6IGF1dG87IiB2aWV3Qm94PSIwIDAgMTAwIDEwMCIgcHJlc2VydmVBc3BlY3RSYXRpbz0ieE1pZFlNaWQiPiAgPGNpcmNsZSBjeD0iNTAiIGN5PSI1MCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjYzNjM2MzIiBzdHJva2Utd2lkdGg9IjEwIiByPSIzNSIgc3Ryb2tlLWRhc2hhcnJheT0iMTY0LjkzMzYxNDMxMzQ2NDE1IDU2Ljk3Nzg3MTQzNzgyMTM4Ij4gICAgPGFuaW1hdGVUcmFuc2Zvcm0gYXR0cmlidXRlTmFtZT0idHJhbnNmb3JtIiB0eXBlPSJyb3RhdGUiIHJlcGVhdENvdW50PSJpbmRlZmluaXRlIiBkdXI9IjFzIiB2YWx1ZXM9IjAgNTAgNTA7MzYwIDUwIDUwIiBrZXlUaW1lcz0iMDsxIj48L2FuaW1hdGVUcmFuc2Zvcm0+ICA8L2NpcmNsZT48L3N2Zz4=\");\n  background-size: auto calc(min(50%, 400px));\n}",{"id":"p4998"},{"id":"p4996"},{"id":"p4997"}],"margin":0,"sizing_mode":"stretch_width","align":"start"}},{"type":"object","name":"GridPlot","id":"p5101","attributes":{"rows":null,"cols":null,"toolbar":{"type":"object","name":"Toolbar","id":"p5109","attributes":{"tools":[{"type":"object","name":"ToolProxy","id":"p5103","attributes":{"tools":[{"type":"object","name":"WheelZoomTool","id":"p5005","attributes":{"tags":["hv_created"],"zoom_together":"none"}},{"type":"object","name":"WheelZoomTool","id":"p5051","attributes":{"tags":["hv_created"],"zoom_together":"none"}}]}},{"type":"object","name":"ToolProxy","id":"p5104","attributes":{"tools":[{"type":"object","name":"HoverTool","id":"p5006","attributes":{"tags":["hv_created"],"renderers":[{"type":"object","name":"GlyphRenderer","id":"p5043","attributes":{"data_source":{"type":"object","name":"ColumnDataSource","id":"p5034","attributes":{"selected":{"type":"object","name":"Selection","id":"p5035","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"p5036"},"data":{"type":"map","entries":[["k",{"type":"ndarray","array":{"type":"bytes","data":"AQAAAAIAAAADAAAABAAAAAUAAAAGAAAABwAAAAgAAAAJAAAACgAAAA=="},"shape":[10],"dtype":"int32","order":"little"}],["inertia",{"type":"ndarray","array":{"type":"bytes","data":"AAAAAADwcUDlwndVTNJoQOCRvNowzF5AGO6ulG/BU0BnEQsuWlNQQOdykvoHDUpAmOLlEoMER0DEjt+KdKRCQIIPyVCpFEBAFoNuwkMTPEA="},"shape":[10],"dtype":"float64","order":"little"}]]}}},"view":{"type":"object","name":"CDSView","id":"p5044","attributes":{"filter":{"type":"object","name":"AllIndices","id":"p5045"}}},"glyph":{"type":"object","name":"Line","id":"p5040","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"k"},"y":{"type":"field","field":"inertia"},"line_color":"#30a2da","line_width":2}},"selection_glyph":{"type":"object","name":"Line","id":"p5046","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"k"},"y":{"type":"field","field":"inertia"},"line_color":"#30a2da","line_width":2}},"nonselection_glyph":{"type":"object","name":"Line","id":"p5041","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"k"},"y":{"type":"field","field":"inertia"},"line_color":"#30a2da","line_alpha":0.1,"line_width":2}},"muted_glyph":{"type":"object","name":"Line","id":"p5042","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"k"},"y":{"type":"field","field":"inertia"},"line_color":"#30a2da","line_alpha":0.2,"line_width":2}}}}],"tooltips":[["k","@{k}"],["inertia","@{inertia}"]]}},{"type":"object","name":"HoverTool","id":"p5052","attributes":{"tags":["hv_created"],"renderers":[{"type":"object","name":"GlyphRenderer","id":"p5089","attributes":{"data_source":{"type":"object","name":"ColumnDataSource","id":"p5080","attributes":{"selected":{"type":"object","name":"Selection","id":"p5081","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"p5082"},"data":{"type":"map","entries":[["k",{"type":"ndarray","array":{"type":"bytes","data":"AQAAAAIAAAADAAAABAAAAAUAAAAGAAAABwAAAAgAAAAJAAAACgAAAA=="},"shape":[10],"dtype":"int32","order":"little"}],["inertia",{"type":"ndarray","array":{"type":"bytes","data":"HOIxWdABcUCbsYa8YLFmQIwHLxP96lhAZkpm+1udR0DoqLMVBwlCQKSOjXW6UTpAD5lBlbj/M0C67ncrjJwwQGaPv/jPKCpAJkACOW0bJEA="},"shape":[10],"dtype":"float64","order":"little"}]]}}},"view":{"type":"object","name":"CDSView","id":"p5090","attributes":{"filter":{"type":"object","name":"AllIndices","id":"p5091"}}},"glyph":{"type":"object","name":"Line","id":"p5086","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"k"},"y":{"type":"field","field":"inertia"},"line_color":"#30a2da","line_width":2}},"selection_glyph":{"type":"object","name":"Line","id":"p5092","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"k"},"y":{"type":"field","field":"inertia"},"line_color":"#30a2da","line_width":2}},"nonselection_glyph":{"type":"object","name":"Line","id":"p5087","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"k"},"y":{"type":"field","field":"inertia"},"line_color":"#30a2da","line_alpha":0.1,"line_width":2}},"muted_glyph":{"type":"object","name":"Line","id":"p5088","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"k"},"y":{"type":"field","field":"inertia"},"line_color":"#30a2da","line_alpha":0.2,"line_width":2}}}}],"tooltips":[["k","@{k}"],["inertia","@{inertia}"]]}}]}},{"type":"object","name":"SaveTool","id":"p5105"},{"type":"object","name":"ToolProxy","id":"p5106","attributes":{"tools":[{"type":"object","name":"PanTool","id":"p5030"},{"type":"object","name":"PanTool","id":"p5076"}]}},{"type":"object","name":"ToolProxy","id":"p5107","attributes":{"tools":[{"type":"object","name":"BoxZoomTool","id":"p5031","attributes":{"overlay":{"type":"object","name":"BoxAnnotation","id":"p5032","attributes":{"syncable":false,"level":"overlay","visible":false,"left_units":"canvas","right_units":"canvas","bottom_units":"canvas","top_units":"canvas","line_color":"black","line_alpha":1.0,"line_width":2,"line_dash":[4,4],"fill_color":"lightgrey","fill_alpha":0.5}}}},{"type":"object","name":"BoxZoomTool","id":"p5077","attributes":{"overlay":{"type":"object","name":"BoxAnnotation","id":"p5078","attributes":{"syncable":false,"level":"overlay","visible":false,"left_units":"canvas","right_units":"canvas","bottom_units":"canvas","top_units":"canvas","line_color":"black","line_alpha":1.0,"line_width":2,"line_dash":[4,4],"fill_color":"lightgrey","fill_alpha":0.5}}}}]}},{"type":"object","name":"ToolProxy","id":"p5108","attributes":{"tools":[{"type":"object","name":"ResetTool","id":"p5033"},{"type":"object","name":"ResetTool","id":"p5079"}]}}]}},"children":[[{"type":"object","name":"Figure","id":"p5007","attributes":{"width":700,"height":300,"sizing_mode":"fixed","align":"start","x_range":{"type":"object","name":"Range1d","id":"p5000","attributes":{"tags":[[["k","k",null]],[]],"start":1.0,"end":10.0,"reset_start":1.0,"reset_end":10.0}},"y_range":{"type":"object","name":"Range1d","id":"p5001","attributes":{"tags":[[["inertia","inertia",null]],{"type":"map","entries":[["invert_yaxis",false],["autorange",false]]}],"start":-17.64107555420189,"end":314.6946432322002,"reset_start":-17.64107555420189,"reset_end":314.6946432322002}},"x_scale":{"type":"object","name":"LinearScale","id":"p5017"},"y_scale":{"type":"object","name":"LinearScale","id":"p5018"},"title":{"type":"object","name":"Title","id":"p5010","attributes":{"text":"Elbow Curve","text_color":"black","text_font_size":"12pt"}},"renderers":[{"id":"p5043"}],"toolbar":{"type":"object","name":"Toolbar","id":"p5016","attributes":{"tools":[{"id":"p5005"},{"id":"p5006"},{"type":"object","name":"SaveTool","id":"p5029"},{"id":"p5030"},{"id":"p5031"},{"id":"p5033"}],"active_drag":{"id":"p5030"},"active_scroll":{"id":"p5005"}}},"toolbar_location":null,"left":[{"type":"object","name":"LinearAxis","id":"p5024","attributes":{"ticker":{"type":"object","name":"BasicTicker","id":"p5025","attributes":{"mantissas":[1,2,5]}},"formatter":{"type":"object","name":"BasicTickFormatter","id":"p5026"},"axis_label":"inertia","major_label_policy":{"type":"object","name":"AllLabels","id":"p5027"}}}],"below":[{"type":"object","name":"LinearAxis","id":"p5019","attributes":{"ticker":{"type":"object","name":"FixedTicker","id":"p5047","attributes":{"ticks":[1,2,3,4,5,6,7,8,9,10],"minor_ticks":[]}},"formatter":{"type":"object","name":"BasicTickFormatter","id":"p5021"},"axis_label":"k","major_label_policy":{"type":"object","name":"AllLabels","id":"p5022"}}}],"center":[{"type":"object","name":"Grid","id":"p5023","attributes":{"axis":{"id":"p5019"},"grid_line_color":null}},{"type":"object","name":"Grid","id":"p5028","attributes":{"dimension":1,"axis":{"id":"p5024"},"grid_line_color":null}}],"min_border_top":10,"min_border_bottom":10,"min_border_left":10,"min_border_right":10,"output_backend":"webgl"}},0,0],[{"type":"object","name":"Figure","id":"p5053","attributes":{"width":700,"height":300,"sizing_mode":"fixed","align":"start","x_range":{"id":"p5000"},"y_range":{"id":"p5001"},"x_scale":{"type":"object","name":"LinearScale","id":"p5063"},"y_scale":{"type":"object","name":"LinearScale","id":"p5064"},"title":{"type":"object","name":"Title","id":"p5056","attributes":{"text":"Elbow Curve","text_color":"black","text_font_size":"12pt"}},"renderers":[{"id":"p5089"}],"toolbar":{"type":"object","name":"Toolbar","id":"p5062","attributes":{"tools":[{"id":"p5051"},{"id":"p5052"},{"type":"object","name":"SaveTool","id":"p5075"},{"id":"p5076"},{"id":"p5077"},{"id":"p5079"}],"active_drag":{"id":"p5076"},"active_scroll":{"id":"p5051"}}},"toolbar_location":null,"left":[{"type":"object","name":"LinearAxis","id":"p5070","attributes":{"ticker":{"type":"object","name":"BasicTicker","id":"p5071","attributes":{"mantissas":[1,2,5]}},"formatter":{"type":"object","name":"BasicTickFormatter","id":"p5072"},"axis_label":"inertia","major_label_policy":{"type":"object","name":"AllLabels","id":"p5073"}}}],"below":[{"type":"object","name":"LinearAxis","id":"p5065","attributes":{"ticker":{"type":"object","name":"FixedTicker","id":"p5093","attributes":{"ticks":[1,2,3,4,5,6,7,8,9,10],"minor_ticks":[]}},"formatter":{"type":"object","name":"BasicTickFormatter","id":"p5067"},"axis_label":"k","major_label_policy":{"type":"object","name":"AllLabels","id":"p5068"}}}],"center":[{"type":"object","name":"Grid","id":"p5069","attributes":{"axis":{"id":"p5065"},"grid_line_color":null}},{"type":"object","name":"Grid","id":"p5074","attributes":{"dimension":1,"axis":{"id":"p5070"},"grid_line_color":null}}],"min_border_top":10,"min_border_bottom":10,"min_border_left":10,"min_border_right":10,"output_backend":"webgl"}},0,1]]}},{"type":"object","name":"Spacer","id":"p5111","attributes":{"name":"HSpacer10600","stylesheets":["\n:host(.pn-loading.pn-arc):before, .pn-loading.pn-arc:before {\n  background-image: url(\"data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHN0eWxlPSJtYXJnaW46IGF1dG87IGJhY2tncm91bmQ6IG5vbmU7IGRpc3BsYXk6IGJsb2NrOyBzaGFwZS1yZW5kZXJpbmc6IGF1dG87IiB2aWV3Qm94PSIwIDAgMTAwIDEwMCIgcHJlc2VydmVBc3BlY3RSYXRpbz0ieE1pZFlNaWQiPiAgPGNpcmNsZSBjeD0iNTAiIGN5PSI1MCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjYzNjM2MzIiBzdHJva2Utd2lkdGg9IjEwIiByPSIzNSIgc3Ryb2tlLWRhc2hhcnJheT0iMTY0LjkzMzYxNDMxMzQ2NDE1IDU2Ljk3Nzg3MTQzNzgyMTM4Ij4gICAgPGFuaW1hdGVUcmFuc2Zvcm0gYXR0cmlidXRlTmFtZT0idHJhbnNmb3JtIiB0eXBlPSJyb3RhdGUiIHJlcGVhdENvdW50PSJpbmRlZmluaXRlIiBkdXI9IjFzIiB2YWx1ZXM9IjAgNTAgNTA7MzYwIDUwIDUwIiBrZXlUaW1lcz0iMDsxIj48L2FuaW1hdGVUcmFuc2Zvcm0+ICA8L2NpcmNsZT48L3N2Zz4=\");\n  background-size: auto calc(min(50%, 400px));\n}",{"id":"p4998"},{"id":"p4996"},{"id":"p4997"}],"margin":0,"sizing_mode":"stretch_width","align":"start"}}]}}],"defs":[{"type":"model","name":"ReactiveHTML1"},{"type":"model","name":"FlexBox1","properties":[{"name":"align_content","kind":"Any","default":"flex-start"},{"name":"align_items","kind":"Any","default":"flex-start"},{"name":"flex_direction","kind":"Any","default":"row"},{"name":"flex_wrap","kind":"Any","default":"wrap"},{"name":"justify_content","kind":"Any","default":"flex-start"}]},{"type":"model","name":"FloatPanel1","properties":[{"name":"config","kind":"Any","default":{"type":"map"}},{"name":"contained","kind":"Any","default":true},{"name":"position","kind":"Any","default":"right-top"},{"name":"offsetx","kind":"Any","default":null},{"name":"offsety","kind":"Any","default":null},{"name":"theme","kind":"Any","default":"primary"},{"name":"status","kind":"Any","default":"normalized"}]},{"type":"model","name":"GridStack1","properties":[{"name":"mode","kind":"Any","default":"warn"},{"name":"ncols","kind":"Any","default":null},{"name":"nrows","kind":"Any","default":null},{"name":"allow_resize","kind":"Any","default":true},{"name":"allow_drag","kind":"Any","default":true},{"name":"state","kind":"Any","default":[]}]},{"type":"model","name":"drag1","properties":[{"name":"slider_width","kind":"Any","default":5},{"name":"slider_color","kind":"Any","default":"black"},{"name":"value","kind":"Any","default":50}]},{"type":"model","name":"click1","properties":[{"name":"terminal_output","kind":"Any","default":""},{"name":"debug_name","kind":"Any","default":""},{"name":"clears","kind":"Any","default":0}]},{"type":"model","name":"FastWrapper1","properties":[{"name":"object","kind":"Any","default":null},{"name":"style","kind":"Any","default":null}]},{"type":"model","name":"NotificationAreaBase1","properties":[{"name":"js_events","kind":"Any","default":{"type":"map"}},{"name":"position","kind":"Any","default":"bottom-right"},{"name":"_clear","kind":"Any","default":0}]},{"type":"model","name":"NotificationArea1","properties":[{"name":"js_events","kind":"Any","default":{"type":"map"}},{"name":"notifications","kind":"Any","default":[]},{"name":"position","kind":"Any","default":"bottom-right"},{"name":"_clear","kind":"Any","default":0},{"name":"types","kind":"Any","default":[{"type":"map","entries":[["type","warning"],["background","#ffc107"],["icon",{"type":"map","entries":[["className","fas fa-exclamation-triangle"],["tagName","i"],["color","white"]]}]]},{"type":"map","entries":[["type","info"],["background","#007bff"],["icon",{"type":"map","entries":[["className","fas fa-info-circle"],["tagName","i"],["color","white"]]}]]}]}]},{"type":"model","name":"Notification","properties":[{"name":"background","kind":"Any","default":null},{"name":"duration","kind":"Any","default":3000},{"name":"icon","kind":"Any","default":null},{"name":"message","kind":"Any","default":""},{"name":"notification_type","kind":"Any","default":null},{"name":"_destroyed","kind":"Any","default":false}]},{"type":"model","name":"TemplateActions1","properties":[{"name":"open_modal","kind":"Any","default":0},{"name":"close_modal","kind":"Any","default":0}]},{"type":"model","name":"BootstrapTemplateActions1","properties":[{"name":"open_modal","kind":"Any","default":0},{"name":"close_modal","kind":"Any","default":0}]},{"type":"model","name":"MaterialTemplateActions1","properties":[{"name":"open_modal","kind":"Any","default":0},{"name":"close_modal","kind":"Any","default":0}]}]}};
  var render_items = [{"docid":"3a2e5fee-2e04-40b9-b709-abc4174a3352","roots":{"p4995":"a2d8d0a7-fbf2-47b1-be38-6ec4c532f6be"},"root_ids":["p4995"]}];
  var docs = Object.values(docs_json)
  if (!docs) {
    return
  }
  const py_version = docs[0].version.replace('rc', '-rc.').replace('.dev', '-dev.')
  const is_dev = py_version.indexOf("+") !== -1 || py_version.indexOf("-") !== -1
  function embed_document(root) {
    var Bokeh = get_bokeh(root)
    Bokeh.embed.embed_items_notebook(docs_json, render_items);
    for (const render_item of render_items) {
      for (const root_id of render_item.root_ids) {
	const id_el = document.getElementById(root_id)
	if (id_el.children.length && (id_el.children[0].className === 'bk-root')) {
	  const root_el = id_el.children[0]
	  root_el.id = root_el.id + '-rendered'
	}
      }
    }
  }
  function get_bokeh(root) {
    if (root.Bokeh === undefined) {
      return null
    } else if (root.Bokeh.version !== py_version && !is_dev) {
      if (root.Bokeh.versions === undefined || !root.Bokeh.versions.has(py_version)) {
	return null
      }
      return root.Bokeh.versions.get(py_version);
    } else if (root.Bokeh.version === py_version) {
      return root.Bokeh
    }
    return null
  }
  function is_loaded(root) {
    var Bokeh = get_bokeh(root)
    return (Bokeh != null && Bokeh.Panel !== undefined)
  }
  if (is_loaded(root)) {
    embed_document(root);
  } else {
    var attempts = 0;
    var timer = setInterval(function(root) {
      if (is_loaded(root)) {
        clearInterval(timer);
        embed_document(root);
      } else if (document.readyState == "complete") {
        attempts++;
        if (attempts > 200) {
          clearInterval(timer);
	  var Bokeh = get_bokeh(root)
	  if (Bokeh == null || Bokeh.Panel == null) {
            console.warn("Panel: ERROR: Unable to run Panel code because Bokeh or Panel library is missing");
	  } else {
	    console.warn("Panel: WARNING: Attempting to render but not all required libraries could be resolved.")
	    embed_document(root)
	  }
        }
      }
    }, 25, root)
  }
})(window);</script>




```python
# Compoosite plot to contrast the clusters
df_scaled_market_pca.hvplot.scatter(
    x = "price_change_percentage_24h",
    y = "price_change_percentage_7d",
    by = "predicted clusters",
    hover_cols = "coin_id",
    title = "Segment Clusters",
    xlabel = "Price Change Percentage 24 Hours",
    ylabel = "Price Change Percentage 7 Days"
) + df_market_data_scaled.hvplot.scatter(
    x = "price_change_percentage_24h",
    y = "price_change_percentage_7d",
    by = "Predicted Clusters",
    hover_cols = "coin_id",
   title = "Segement Clusters",
    xlabel = "Price Change Percentage 24 Hours",
    ylabel = "Price Change Percentage 7 Days",    
)
```






<div id='p4748'>
  <div id="f7ae49d1-5411-4876-99f9-bcee215b4502" data-root-id="p4748" style="display: contents;"></div>
</div>
<script type="application/javascript">(function(root) {
  var docs_json = {"479ec3f4-1c8e-4a97-86bf-0fcce7323cce":{"version":"3.2.1","title":"Bokeh Application","roots":[{"type":"object","name":"Row","id":"p4748","attributes":{"name":"Row10108","tags":["embedded"],"stylesheets":["\n:host(.pn-loading.pn-arc):before, .pn-loading.pn-arc:before {\n  background-image: url(\"data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHN0eWxlPSJtYXJnaW46IGF1dG87IGJhY2tncm91bmQ6IG5vbmU7IGRpc3BsYXk6IGJsb2NrOyBzaGFwZS1yZW5kZXJpbmc6IGF1dG87IiB2aWV3Qm94PSIwIDAgMTAwIDEwMCIgcHJlc2VydmVBc3BlY3RSYXRpbz0ieE1pZFlNaWQiPiAgPGNpcmNsZSBjeD0iNTAiIGN5PSI1MCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjYzNjM2MzIiBzdHJva2Utd2lkdGg9IjEwIiByPSIzNSIgc3Ryb2tlLWRhc2hhcnJheT0iMTY0LjkzMzYxNDMxMzQ2NDE1IDU2Ljk3Nzg3MTQzNzgyMTM4Ij4gICAgPGFuaW1hdGVUcmFuc2Zvcm0gYXR0cmlidXRlTmFtZT0idHJhbnNmb3JtIiB0eXBlPSJyb3RhdGUiIHJlcGVhdENvdW50PSJpbmRlZmluaXRlIiBkdXI9IjFzIiB2YWx1ZXM9IjAgNTAgNTA7MzYwIDUwIDUwIiBrZXlUaW1lcz0iMDsxIj48L2FuaW1hdGVUcmFuc2Zvcm0+ICA8L2NpcmNsZT48L3N2Zz4=\");\n  background-size: auto calc(min(50%, 400px));\n}",{"type":"object","name":"ImportedStyleSheet","id":"p4751","attributes":{"url":"https://cdn.holoviz.org/panel/1.2.1/dist/css/loading.css"}},{"type":"object","name":"ImportedStyleSheet","id":"p4977","attributes":{"url":"https://cdn.holoviz.org/panel/1.2.1/dist/css/listpanel.css"}},{"type":"object","name":"ImportedStyleSheet","id":"p4749","attributes":{"url":"https://cdn.holoviz.org/panel/1.2.1/dist/bundled/theme/default.css"}},{"type":"object","name":"ImportedStyleSheet","id":"p4750","attributes":{"url":"https://cdn.holoviz.org/panel/1.2.1/dist/bundled/theme/native.css"}}],"margin":0,"sizing_mode":"stretch_width","align":"start","children":[{"type":"object","name":"Spacer","id":"p4752","attributes":{"name":"HSpacer10118","stylesheets":["\n:host(.pn-loading.pn-arc):before, .pn-loading.pn-arc:before {\n  background-image: url(\"data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHN0eWxlPSJtYXJnaW46IGF1dG87IGJhY2tncm91bmQ6IG5vbmU7IGRpc3BsYXk6IGJsb2NrOyBzaGFwZS1yZW5kZXJpbmc6IGF1dG87IiB2aWV3Qm94PSIwIDAgMTAwIDEwMCIgcHJlc2VydmVBc3BlY3RSYXRpbz0ieE1pZFlNaWQiPiAgPGNpcmNsZSBjeD0iNTAiIGN5PSI1MCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjYzNjM2MzIiBzdHJva2Utd2lkdGg9IjEwIiByPSIzNSIgc3Ryb2tlLWRhc2hhcnJheT0iMTY0LjkzMzYxNDMxMzQ2NDE1IDU2Ljk3Nzg3MTQzNzgyMTM4Ij4gICAgPGFuaW1hdGVUcmFuc2Zvcm0gYXR0cmlidXRlTmFtZT0idHJhbnNmb3JtIiB0eXBlPSJyb3RhdGUiIHJlcGVhdENvdW50PSJpbmRlZmluaXRlIiBkdXI9IjFzIiB2YWx1ZXM9IjAgNTAgNTA7MzYwIDUwIDUwIiBrZXlUaW1lcz0iMDsxIj48L2FuaW1hdGVUcmFuc2Zvcm0+ICA8L2NpcmNsZT48L3N2Zz4=\");\n  background-size: auto calc(min(50%, 400px));\n}",{"id":"p4751"},{"id":"p4749"},{"id":"p4750"}],"margin":0,"sizing_mode":"stretch_width","align":"start"}},{"type":"object","name":"GridPlot","id":"p4957","attributes":{"rows":null,"cols":null,"toolbar":{"type":"object","name":"Toolbar","id":"p4973","attributes":{"tools":[{"type":"object","name":"ToolProxy","id":"p4967","attributes":{"tools":[{"type":"object","name":"WheelZoomTool","id":"p4758","attributes":{"tags":["hv_created"],"zoom_together":"none"}},{"type":"object","name":"WheelZoomTool","id":"p4856","attributes":{"tags":["hv_created"],"zoom_together":"none"}}]}},{"type":"object","name":"ToolProxy","id":"p4968","attributes":{"tools":[{"type":"object","name":"HoverTool","id":"p4759","attributes":{"tags":["hv_created"],"renderers":[{"type":"object","name":"GlyphRenderer","id":"p4805","attributes":{"name":"0","js_property_callbacks":{"type":"map","entries":[["change:muted",[{"type":"object","name":"CustomJS","id":"p4959","attributes":{"args":{"type":"map","entries":[["src",{"id":"p4805"}],["dst",{"type":"object","name":"GlyphRenderer","id":"p4903","attributes":{"name":"0","js_property_callbacks":{"type":"map","entries":[["change:muted",[{"type":"object","name":"CustomJS","id":"p4960","attributes":{"args":{"type":"map","entries":[["src",{"id":"p4903"}],["dst",{"id":"p4805"}]]},"code":"dst.muted = src.muted"}}]]]},"data_source":{"type":"object","name":"ColumnDataSource","id":"p4894","attributes":{"selected":{"type":"object","name":"Selection","id":"p4895","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"p4896"},"data":{"type":"map","entries":[["price_change_percentage_24h",{"type":"ndarray","array":{"type":"bytes","data":"mCJF1OlLlj/mJRG3Ed+kv1qZGRifu9w//rJXHrAz1T/yMTqenpahPx9kk3dK7sM/GTRehHalwD90fnl54Irtv9XPDahDw9G/+VOWQSEmxz+kpbukw1bSP6eM4M4OgKo/BcMOj9+jqr/Chq6S5ebLv2h3TIfCZ68/weEQ8QyV0D/KmZTmC7niP1YH9WGFy/m/N7JuVBEB07/EU4GAhEGyv8Ocf0hw2/a/lrPBJY23/j/IKcQzxy3av5WrmDyeito/Bf22eE/6sz9Y362Ir3rzPw=="},"shape":[26],"dtype":"float64","order":"little"}],["price_change_percentage_7d",{"type":"ndarray","array":{"type":"bytes","data":"bZSIvk+a5r9RujxqH/Ppvw2U7adXaMi/d+r7VGbW+b9EGSx/8nTnv4rfBo4Lhe2/BEBwXEEApb8Udj8VbP/2vySelkVDp9i/7SZvVsyO5r8uu+JVH+LUvziN+gRMte2/o2qlez1D3b/CtPLKIS7rvyYjaswHnea/US6joeTvzz/dRmuCvtDvv/9TV66U6fq/ZWX+mWVCuD8GJ0pHuV/Nv8nIE2ZWH5q/TTxj/me11z+Htdvf1gXtv2Avihmzf9o/e31xtwIC5r/STlu6Y3Ljvw=="},"shape":[26],"dtype":"float64","order":"little"}],["coin_id",["tether","ripple","bitcoin-cash-sv","crypto-com-chain","usd-coin","eos","tron","okb","stellar","cdai","neo","leo-token","huobi-token","nem","binance-usd","iota","vechain","theta-token","dash","ethereum-classic","havven","omisego","ontology","ftx-token","true-usd","digibyte"]],["Predicted_Clusters",[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]]}}},"view":{"type":"object","name":"CDSView","id":"p4904","attributes":{"filter":{"type":"object","name":"AllIndices","id":"p4905"}}},"glyph":{"type":"object","name":"Scatter","id":"p4900","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"price_change_percentage_24h"},"y":{"type":"field","field":"price_change_percentage_7d"},"size":{"type":"value","value":5.477225575051661},"line_color":{"type":"value","value":"#30a2da"},"fill_color":{"type":"value","value":"#30a2da"},"hatch_color":{"type":"value","value":"#30a2da"}}},"selection_glyph":{"type":"object","name":"Scatter","id":"p4908","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"price_change_percentage_24h"},"y":{"type":"field","field":"price_change_percentage_7d"},"size":{"type":"value","value":5.477225575051661},"angle":{"type":"value","value":0.0},"line_color":{"type":"value","value":"#30a2da"},"line_alpha":{"type":"value","value":1.0},"line_width":{"type":"value","value":1},"line_join":{"type":"value","value":"bevel"},"line_cap":{"type":"value","value":"butt"},"line_dash":{"type":"value","value":[]},"line_dash_offset":{"type":"value","value":0},"fill_color":{"type":"value","value":"#30a2da"},"fill_alpha":{"type":"value","value":1.0},"hatch_color":{"type":"value","value":"#30a2da"},"hatch_alpha":{"type":"value","value":1.0},"hatch_scale":{"type":"value","value":12.0},"hatch_pattern":{"type":"value","value":null},"hatch_weight":{"type":"value","value":1.0},"marker":{"type":"value","value":"circle"}}},"nonselection_glyph":{"type":"object","name":"Scatter","id":"p4901","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"price_change_percentage_24h"},"y":{"type":"field","field":"price_change_percentage_7d"},"size":{"type":"value","value":5.477225575051661},"line_color":{"type":"value","value":"#30a2da"},"line_alpha":{"type":"value","value":0.1},"fill_color":{"type":"value","value":"#30a2da"},"fill_alpha":{"type":"value","value":0.1},"hatch_color":{"type":"value","value":"#30a2da"},"hatch_alpha":{"type":"value","value":0.1}}},"muted_glyph":{"type":"object","name":"Scatter","id":"p4902","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"price_change_percentage_24h"},"y":{"type":"field","field":"price_change_percentage_7d"},"size":{"type":"value","value":5.477225575051661},"line_color":{"type":"value","value":"#30a2da"},"line_alpha":{"type":"value","value":0.2},"fill_color":{"type":"value","value":"#30a2da"},"fill_alpha":{"type":"value","value":0.2},"hatch_color":{"type":"value","value":"#30a2da"},"hatch_alpha":{"type":"value","value":0.2}}}}}]]},"code":"dst.muted = src.muted"}}]]]},"data_source":{"type":"object","name":"ColumnDataSource","id":"p4796","attributes":{"selected":{"type":"object","name":"Selection","id":"p4797","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"p4798"},"data":{"type":"map","entries":[["price_change_percentage_24h",{"type":"ndarray","array":{"type":"bytes","data":"PYEWfZ9Luj/igSoNYqOzP7jRiMaxetI/Fgrh4gxS3D+NasEmYpDOPysFYCd8LtU/y0hran0Gvr8jAPanmaH6Pw3tdw6zK+S/AKU1OyJR1z89qaf9yFq3P+HxjfQqScA/nqxLEE6Io78="},"shape":[13],"dtype":"float64","order":"little"}],["price_change_percentage_7d",{"type":"ndarray","array":{"type":"bytes","data":"urMJzq588b9h/uLSjI3mvwvgJ23+fgPAqC/ykVJc+L802JJu0Zzpv2B/Eu+ciuy/+Ck8Uu2u6L+evjykvVMEwGWKJpxGzbG/8KBlSoRY67+xyOCyUEzxv31mpzwdCue/LwIt7iXp2b8="},"shape":[13],"dtype":"float64","order":"little"}],["coin_id",["bitcoin","ethereum","bitcoin-cash","binancecoin","chainlink","cardano","litecoin","monero","tezos","cosmos","wrapped-bitcoin","zcash","maker"]],["predicted_clusters",[0,0,0,0,0,0,0,0,0,0,0,0,0]]]}}},"view":{"type":"object","name":"CDSView","id":"p4806","attributes":{"filter":{"type":"object","name":"AllIndices","id":"p4807"}}},"glyph":{"type":"object","name":"Scatter","id":"p4802","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"price_change_percentage_24h"},"y":{"type":"field","field":"price_change_percentage_7d"},"size":{"type":"value","value":5.477225575051661},"line_color":{"type":"value","value":"#30a2da"},"fill_color":{"type":"value","value":"#30a2da"},"hatch_color":{"type":"value","value":"#30a2da"}}},"selection_glyph":{"type":"object","name":"Scatter","id":"p4810","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"price_change_percentage_24h"},"y":{"type":"field","field":"price_change_percentage_7d"},"size":{"type":"value","value":5.477225575051661},"angle":{"type":"value","value":0.0},"line_color":{"type":"value","value":"#30a2da"},"line_alpha":{"type":"value","value":1.0},"line_width":{"type":"value","value":1},"line_join":{"type":"value","value":"bevel"},"line_cap":{"type":"value","value":"butt"},"line_dash":{"type":"value","value":[]},"line_dash_offset":{"type":"value","value":0},"fill_color":{"type":"value","value":"#30a2da"},"fill_alpha":{"type":"value","value":1.0},"hatch_color":{"type":"value","value":"#30a2da"},"hatch_alpha":{"type":"value","value":1.0},"hatch_scale":{"type":"value","value":12.0},"hatch_pattern":{"type":"value","value":null},"hatch_weight":{"type":"value","value":1.0},"marker":{"type":"value","value":"circle"}}},"nonselection_glyph":{"type":"object","name":"Scatter","id":"p4803","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"price_change_percentage_24h"},"y":{"type":"field","field":"price_change_percentage_7d"},"size":{"type":"value","value":5.477225575051661},"line_color":{"type":"value","value":"#30a2da"},"line_alpha":{"type":"value","value":0.1},"fill_color":{"type":"value","value":"#30a2da"},"fill_alpha":{"type":"value","value":0.1},"hatch_color":{"type":"value","value":"#30a2da"},"hatch_alpha":{"type":"value","value":0.1}}},"muted_glyph":{"type":"object","name":"Scatter","id":"p4804","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"price_change_percentage_24h"},"y":{"type":"field","field":"price_change_percentage_7d"},"size":{"type":"value","value":5.477225575051661},"line_color":{"type":"value","value":"#30a2da"},"line_alpha":{"type":"value","value":0.2},"fill_color":{"type":"value","value":"#30a2da"},"fill_alpha":{"type":"value","value":0.2},"hatch_color":{"type":"value","value":"#30a2da"},"hatch_alpha":{"type":"value","value":0.2}}}}},{"type":"object","name":"GlyphRenderer","id":"p4820","attributes":{"name":"1","js_property_callbacks":{"type":"map","entries":[["change:muted",[{"type":"object","name":"CustomJS","id":"p4961","attributes":{"args":{"type":"map","entries":[["src",{"id":"p4820"}],["dst",{"type":"object","name":"GlyphRenderer","id":"p4918","attributes":{"name":"1","js_property_callbacks":{"type":"map","entries":[["change:muted",[{"type":"object","name":"CustomJS","id":"p4962","attributes":{"args":{"type":"map","entries":[["src",{"id":"p4918"}],["dst",{"id":"p4820"}]]},"code":"dst.muted = src.muted"}}]]]},"data_source":{"type":"object","name":"ColumnDataSource","id":"p4909","attributes":{"selected":{"type":"object","name":"Selection","id":"p4910","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"p4911"},"data":{"type":"map","entries":[["price_change_percentage_24h",{"type":"ndarray","array":{"type":"bytes","data":"DdlqYN9F4D86azDosLzHPzALIwCtFvM/9w6McDSK7D/VDHOpb1eHPzMEvaFkP7o/Sds39trWsz9Epj8/c9DQP4xAeNEVZ8O/qq8st/Fh0L8FWWk3mH7gP6wo083aUMC//ACFOIMFwL8="},"shape":[13],"dtype":"float64","order":"little"}],["price_change_percentage_7d",{"type":"ndarray","array":{"type":"bytes","data":"U1k8q3mQ3z9yoPpI+ebtPzAB2dP2AQBAgmkOMZk89T8FNulI+JMEQCZRaSLGIPg/84wjaB1l1T8rMp3jf678P2VmURqKqeY/XK6ZksNx/T9DfMqy1o7dPyTPDB1Xu+0/srW7TleV4j8="},"shape":[13],"dtype":"float64","order":"little"}],["coin_id",["bitcoin","ethereum","bitcoin-cash","binancecoin","chainlink","cardano","litecoin","monero","tezos","cosmos","wrapped-bitcoin","zcash","maker"]],["Predicted_Clusters",[1,1,1,1,1,1,1,1,1,1,1,1,1]]]}}},"view":{"type":"object","name":"CDSView","id":"p4919","attributes":{"filter":{"type":"object","name":"AllIndices","id":"p4920"}}},"glyph":{"type":"object","name":"Scatter","id":"p4915","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"price_change_percentage_24h"},"y":{"type":"field","field":"price_change_percentage_7d"},"size":{"type":"value","value":5.477225575051661},"line_color":{"type":"value","value":"#fc4f30"},"fill_color":{"type":"value","value":"#fc4f30"},"hatch_color":{"type":"value","value":"#fc4f30"}}},"selection_glyph":{"type":"object","name":"Scatter","id":"p4922","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"price_change_percentage_24h"},"y":{"type":"field","field":"price_change_percentage_7d"},"size":{"type":"value","value":5.477225575051661},"angle":{"type":"value","value":0.0},"line_color":{"type":"value","value":"#fc4f30"},"line_alpha":{"type":"value","value":1.0},"line_width":{"type":"value","value":1},"line_join":{"type":"value","value":"bevel"},"line_cap":{"type":"value","value":"butt"},"line_dash":{"type":"value","value":[]},"line_dash_offset":{"type":"value","value":0},"fill_color":{"type":"value","value":"#fc4f30"},"fill_alpha":{"type":"value","value":1.0},"hatch_color":{"type":"value","value":"#fc4f30"},"hatch_alpha":{"type":"value","value":1.0},"hatch_scale":{"type":"value","value":12.0},"hatch_pattern":{"type":"value","value":null},"hatch_weight":{"type":"value","value":1.0},"marker":{"type":"value","value":"circle"}}},"nonselection_glyph":{"type":"object","name":"Scatter","id":"p4916","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"price_change_percentage_24h"},"y":{"type":"field","field":"price_change_percentage_7d"},"size":{"type":"value","value":5.477225575051661},"line_color":{"type":"value","value":"#fc4f30"},"line_alpha":{"type":"value","value":0.1},"fill_color":{"type":"value","value":"#fc4f30"},"fill_alpha":{"type":"value","value":0.1},"hatch_color":{"type":"value","value":"#fc4f30"},"hatch_alpha":{"type":"value","value":0.1}}},"muted_glyph":{"type":"object","name":"Scatter","id":"p4917","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"price_change_percentage_24h"},"y":{"type":"field","field":"price_change_percentage_7d"},"size":{"type":"value","value":5.477225575051661},"line_color":{"type":"value","value":"#fc4f30"},"line_alpha":{"type":"value","value":0.2},"fill_color":{"type":"value","value":"#fc4f30"},"fill_alpha":{"type":"value","value":0.2},"hatch_color":{"type":"value","value":"#fc4f30"},"hatch_alpha":{"type":"value","value":0.2}}}}}]]},"code":"dst.muted = src.muted"}}]]]},"data_source":{"type":"object","name":"ColumnDataSource","id":"p4811","attributes":{"selected":{"type":"object","name":"Selection","id":"p4812","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"p4813"},"data":{"type":"map","entries":[["price_change_percentage_24h",{"type":"ndarray","array":{"type":"bytes","data":"nGyQQgfX478pSYPDbEXlv3owJkhA6uu/bj7D2hwx8b/nDqtxxSHkvz3/ePHv4e6/QQav7I2X4L+5Rfvqeu7nv8loLIxVUey/O0g9rACf5b/k3WkGGzjuv0IkCmugouu/SuNrVqAx47+2MS8bgh/hPxHWGIfcDOS/QhAmCKHP778GuhAvPB75v0lozcaR0P4/j8PZqVsP678vhJO800fpvxG0ohWcgdy/Sd6OEdarsD8NvGGS39D6v2YnnZwYd+O/Y9l4MCQC5L8ZUo+f6ujgvw=="},"shape":[26],"dtype":"float64","order":"little"}],["price_change_percentage_7d",{"type":"ndarray","array":{"type":"bytes","data":"qKV1G1k+lL+4jSIpJ89hPxu8vi3S4cW/Mpn6aWEA8j9z0tFdaTSQvz1gIz6zdbo/hKMFK4/zsL/rM3NTlf3yP7WyW+gMt9o/mcMX3mETtb8QQNGrt/PmP13UOEjIqts/f96jznRamD+BPCC+xNW3v+uD0FfGCaa/UMsfnHrWtT9DZsFUngj3PwAWcGTTr/c/wXduBhVMwD/x0u4oRh6qP5yTdUvdQgNAvC8ePl3xzb+rscH76IL4P4EqY26iQdK/6lea4uDzrr87fb92BbmyPw=="},"shape":[26],"dtype":"float64","order":"little"}],["coin_id",["tether","ripple","bitcoin-cash-sv","crypto-com-chain","usd-coin","eos","tron","okb","stellar","cdai","neo","leo-token","huobi-token","nem","binance-usd","iota","vechain","theta-token","dash","ethereum-classic","havven","omisego","ontology","ftx-token","true-usd","digibyte"]],["predicted_clusters",[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]]]}}},"view":{"type":"object","name":"CDSView","id":"p4821","attributes":{"filter":{"type":"object","name":"AllIndices","id":"p4822"}}},"glyph":{"type":"object","name":"Scatter","id":"p4817","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"price_change_percentage_24h"},"y":{"type":"field","field":"price_change_percentage_7d"},"size":{"type":"value","value":5.477225575051661},"line_color":{"type":"value","value":"#fc4f30"},"fill_color":{"type":"value","value":"#fc4f30"},"hatch_color":{"type":"value","value":"#fc4f30"}}},"selection_glyph":{"type":"object","name":"Scatter","id":"p4824","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"price_change_percentage_24h"},"y":{"type":"field","field":"price_change_percentage_7d"},"size":{"type":"value","value":5.477225575051661},"angle":{"type":"value","value":0.0},"line_color":{"type":"value","value":"#fc4f30"},"line_alpha":{"type":"value","value":1.0},"line_width":{"type":"value","value":1},"line_join":{"type":"value","value":"bevel"},"line_cap":{"type":"value","value":"butt"},"line_dash":{"type":"value","value":[]},"line_dash_offset":{"type":"value","value":0},"fill_color":{"type":"value","value":"#fc4f30"},"fill_alpha":{"type":"value","value":1.0},"hatch_color":{"type":"value","value":"#fc4f30"},"hatch_alpha":{"type":"value","value":1.0},"hatch_scale":{"type":"value","value":12.0},"hatch_pattern":{"type":"value","value":null},"hatch_weight":{"type":"value","value":1.0},"marker":{"type":"value","value":"circle"}}},"nonselection_glyph":{"type":"object","name":"Scatter","id":"p4818","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"price_change_percentage_24h"},"y":{"type":"field","field":"price_change_percentage_7d"},"size":{"type":"value","value":5.477225575051661},"line_color":{"type":"value","value":"#fc4f30"},"line_alpha":{"type":"value","value":0.1},"fill_color":{"type":"value","value":"#fc4f30"},"fill_alpha":{"type":"value","value":0.1},"hatch_color":{"type":"value","value":"#fc4f30"},"hatch_alpha":{"type":"value","value":0.1}}},"muted_glyph":{"type":"object","name":"Scatter","id":"p4819","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"price_change_percentage_24h"},"y":{"type":"field","field":"price_change_percentage_7d"},"size":{"type":"value","value":5.477225575051661},"line_color":{"type":"value","value":"#fc4f30"},"line_alpha":{"type":"value","value":0.2},"fill_color":{"type":"value","value":"#fc4f30"},"fill_alpha":{"type":"value","value":0.2},"hatch_color":{"type":"value","value":"#fc4f30"},"hatch_alpha":{"type":"value","value":0.2}}}}},{"type":"object","name":"GlyphRenderer","id":"p4834","attributes":{"name":"2","js_property_callbacks":{"type":"map","entries":[["change:muted",[{"type":"object","name":"CustomJS","id":"p4963","attributes":{"args":{"type":"map","entries":[["src",{"id":"p4834"}],["dst",{"type":"object","name":"GlyphRenderer","id":"p4932","attributes":{"name":"2","js_property_callbacks":{"type":"map","entries":[["change:muted",[{"type":"object","name":"CustomJS","id":"p4964","attributes":{"args":{"type":"map","entries":[["src",{"id":"p4932"}],["dst",{"id":"p4834"}]]},"code":"dst.muted = src.muted"}}]]]},"data_source":{"type":"object","name":"ColumnDataSource","id":"p4923","attributes":{"selected":{"type":"object","name":"Selection","id":"p4924","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"p4925"},"data":{"type":"map","entries":[["price_change_percentage_24h",{"type":"ndarray","array":{"type":"bytes","data":"3WeHPpbsE8A="},"shape":[1],"dtype":"float64","order":"little"}],["price_change_percentage_7d",{"type":"ndarray","array":{"type":"bytes","data":"dIWi2pshp78="},"shape":[1],"dtype":"float64","order":"little"}],["coin_id",["ethlend"]],["Predicted_Clusters",[2]]]}}},"view":{"type":"object","name":"CDSView","id":"p4933","attributes":{"filter":{"type":"object","name":"AllIndices","id":"p4934"}}},"glyph":{"type":"object","name":"Scatter","id":"p4929","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"price_change_percentage_24h"},"y":{"type":"field","field":"price_change_percentage_7d"},"size":{"type":"value","value":5.477225575051661},"line_color":{"type":"value","value":"#e5ae38"},"fill_color":{"type":"value","value":"#e5ae38"},"hatch_color":{"type":"value","value":"#e5ae38"}}},"selection_glyph":{"type":"object","name":"Scatter","id":"p4936","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"price_change_percentage_24h"},"y":{"type":"field","field":"price_change_percentage_7d"},"size":{"type":"value","value":5.477225575051661},"angle":{"type":"value","value":0.0},"line_color":{"type":"value","value":"#e5ae38"},"line_alpha":{"type":"value","value":1.0},"line_width":{"type":"value","value":1},"line_join":{"type":"value","value":"bevel"},"line_cap":{"type":"value","value":"butt"},"line_dash":{"type":"value","value":[]},"line_dash_offset":{"type":"value","value":0},"fill_color":{"type":"value","value":"#e5ae38"},"fill_alpha":{"type":"value","value":1.0},"hatch_color":{"type":"value","value":"#e5ae38"},"hatch_alpha":{"type":"value","value":1.0},"hatch_scale":{"type":"value","value":12.0},"hatch_pattern":{"type":"value","value":null},"hatch_weight":{"type":"value","value":1.0},"marker":{"type":"value","value":"circle"}}},"nonselection_glyph":{"type":"object","name":"Scatter","id":"p4930","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"price_change_percentage_24h"},"y":{"type":"field","field":"price_change_percentage_7d"},"size":{"type":"value","value":5.477225575051661},"line_color":{"type":"value","value":"#e5ae38"},"line_alpha":{"type":"value","value":0.1},"fill_color":{"type":"value","value":"#e5ae38"},"fill_alpha":{"type":"value","value":0.1},"hatch_color":{"type":"value","value":"#e5ae38"},"hatch_alpha":{"type":"value","value":0.1}}},"muted_glyph":{"type":"object","name":"Scatter","id":"p4931","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"price_change_percentage_24h"},"y":{"type":"field","field":"price_change_percentage_7d"},"size":{"type":"value","value":5.477225575051661},"line_color":{"type":"value","value":"#e5ae38"},"line_alpha":{"type":"value","value":0.2},"fill_color":{"type":"value","value":"#e5ae38"},"fill_alpha":{"type":"value","value":0.2},"hatch_color":{"type":"value","value":"#e5ae38"},"hatch_alpha":{"type":"value","value":0.2}}}}}]]},"code":"dst.muted = src.muted"}}]]]},"data_source":{"type":"object","name":"ColumnDataSource","id":"p4825","attributes":{"selected":{"type":"object","name":"Selection","id":"p4826","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"p4827"},"data":{"type":"map","entries":[["price_change_percentage_24h",{"type":"ndarray","array":{"type":"bytes","data":"vvRIi0cyFkA="},"shape":[1],"dtype":"float64","order":"little"}],["price_change_percentage_7d",{"type":"ndarray","array":{"type":"bytes","data":"jxm+37gNHUA="},"shape":[1],"dtype":"float64","order":"little"}],["coin_id",["ethlend"]],["predicted_clusters",[2]]]}}},"view":{"type":"object","name":"CDSView","id":"p4835","attributes":{"filter":{"type":"object","name":"AllIndices","id":"p4836"}}},"glyph":{"type":"object","name":"Scatter","id":"p4831","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"price_change_percentage_24h"},"y":{"type":"field","field":"price_change_percentage_7d"},"size":{"type":"value","value":5.477225575051661},"line_color":{"type":"value","value":"#e5ae38"},"fill_color":{"type":"value","value":"#e5ae38"},"hatch_color":{"type":"value","value":"#e5ae38"}}},"selection_glyph":{"type":"object","name":"Scatter","id":"p4838","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"price_change_percentage_24h"},"y":{"type":"field","field":"price_change_percentage_7d"},"size":{"type":"value","value":5.477225575051661},"angle":{"type":"value","value":0.0},"line_color":{"type":"value","value":"#e5ae38"},"line_alpha":{"type":"value","value":1.0},"line_width":{"type":"value","value":1},"line_join":{"type":"value","value":"bevel"},"line_cap":{"type":"value","value":"butt"},"line_dash":{"type":"value","value":[]},"line_dash_offset":{"type":"value","value":0},"fill_color":{"type":"value","value":"#e5ae38"},"fill_alpha":{"type":"value","value":1.0},"hatch_color":{"type":"value","value":"#e5ae38"},"hatch_alpha":{"type":"value","value":1.0},"hatch_scale":{"type":"value","value":12.0},"hatch_pattern":{"type":"value","value":null},"hatch_weight":{"type":"value","value":1.0},"marker":{"type":"value","value":"circle"}}},"nonselection_glyph":{"type":"object","name":"Scatter","id":"p4832","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"price_change_percentage_24h"},"y":{"type":"field","field":"price_change_percentage_7d"},"size":{"type":"value","value":5.477225575051661},"line_color":{"type":"value","value":"#e5ae38"},"line_alpha":{"type":"value","value":0.1},"fill_color":{"type":"value","value":"#e5ae38"},"fill_alpha":{"type":"value","value":0.1},"hatch_color":{"type":"value","value":"#e5ae38"},"hatch_alpha":{"type":"value","value":0.1}}},"muted_glyph":{"type":"object","name":"Scatter","id":"p4833","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"price_change_percentage_24h"},"y":{"type":"field","field":"price_change_percentage_7d"},"size":{"type":"value","value":5.477225575051661},"line_color":{"type":"value","value":"#e5ae38"},"line_alpha":{"type":"value","value":0.2},"fill_color":{"type":"value","value":"#e5ae38"},"fill_alpha":{"type":"value","value":0.2},"hatch_color":{"type":"value","value":"#e5ae38"},"hatch_alpha":{"type":"value","value":0.2}}}}},{"type":"object","name":"GlyphRenderer","id":"p4848","attributes":{"name":"3","js_property_callbacks":{"type":"map","entries":[["change:muted",[{"type":"object","name":"CustomJS","id":"p4965","attributes":{"args":{"type":"map","entries":[["src",{"id":"p4848"}],["dst",{"type":"object","name":"GlyphRenderer","id":"p4946","attributes":{"name":"3","js_property_callbacks":{"type":"map","entries":[["change:muted",[{"type":"object","name":"CustomJS","id":"p4966","attributes":{"args":{"type":"map","entries":[["src",{"id":"p4946"}],["dst",{"id":"p4848"}]]},"code":"dst.muted = src.muted"}}]]]},"data_source":{"type":"object","name":"ColumnDataSource","id":"p4937","attributes":{"selected":{"type":"object","name":"Selection","id":"p4938","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"p4939"},"data":{"type":"map","entries":[["price_change_percentage_24h",{"type":"ndarray","array":{"type":"bytes","data":"VLqXBn668D8="},"shape":[1],"dtype":"float64","order":"little"}],["price_change_percentage_7d",{"type":"ndarray","array":{"type":"bytes","data":"Nv03JFjJ478="},"shape":[1],"dtype":"float64","order":"little"}],["coin_id",["celsius-degree-token"]],["Predicted_Clusters",[3]]]}}},"view":{"type":"object","name":"CDSView","id":"p4947","attributes":{"filter":{"type":"object","name":"AllIndices","id":"p4948"}}},"glyph":{"type":"object","name":"Scatter","id":"p4943","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"price_change_percentage_24h"},"y":{"type":"field","field":"price_change_percentage_7d"},"size":{"type":"value","value":5.477225575051661},"line_color":{"type":"value","value":"#6d904f"},"fill_color":{"type":"value","value":"#6d904f"},"hatch_color":{"type":"value","value":"#6d904f"}}},"selection_glyph":{"type":"object","name":"Scatter","id":"p4950","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"price_change_percentage_24h"},"y":{"type":"field","field":"price_change_percentage_7d"},"size":{"type":"value","value":5.477225575051661},"angle":{"type":"value","value":0.0},"line_color":{"type":"value","value":"#6d904f"},"line_alpha":{"type":"value","value":1.0},"line_width":{"type":"value","value":1},"line_join":{"type":"value","value":"bevel"},"line_cap":{"type":"value","value":"butt"},"line_dash":{"type":"value","value":[]},"line_dash_offset":{"type":"value","value":0},"fill_color":{"type":"value","value":"#6d904f"},"fill_alpha":{"type":"value","value":1.0},"hatch_color":{"type":"value","value":"#6d904f"},"hatch_alpha":{"type":"value","value":1.0},"hatch_scale":{"type":"value","value":12.0},"hatch_pattern":{"type":"value","value":null},"hatch_weight":{"type":"value","value":1.0},"marker":{"type":"value","value":"circle"}}},"nonselection_glyph":{"type":"object","name":"Scatter","id":"p4944","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"price_change_percentage_24h"},"y":{"type":"field","field":"price_change_percentage_7d"},"size":{"type":"value","value":5.477225575051661},"line_color":{"type":"value","value":"#6d904f"},"line_alpha":{"type":"value","value":0.1},"fill_color":{"type":"value","value":"#6d904f"},"fill_alpha":{"type":"value","value":0.1},"hatch_color":{"type":"value","value":"#6d904f"},"hatch_alpha":{"type":"value","value":0.1}}},"muted_glyph":{"type":"object","name":"Scatter","id":"p4945","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"price_change_percentage_24h"},"y":{"type":"field","field":"price_change_percentage_7d"},"size":{"type":"value","value":5.477225575051661},"line_color":{"type":"value","value":"#6d904f"},"line_alpha":{"type":"value","value":0.2},"fill_color":{"type":"value","value":"#6d904f"},"fill_alpha":{"type":"value","value":0.2},"hatch_color":{"type":"value","value":"#6d904f"},"hatch_alpha":{"type":"value","value":0.2}}}}}]]},"code":"dst.muted = src.muted"}}]]]},"data_source":{"type":"object","name":"ColumnDataSource","id":"p4839","attributes":{"selected":{"type":"object","name":"Selection","id":"p4840","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"p4841"},"data":{"type":"map","entries":[["price_change_percentage_24h",{"type":"ndarray","array":{"type":"bytes","data":"+D4Vk/wCH0A="},"shape":[1],"dtype":"float64","order":"little"}],["price_change_percentage_7d",{"type":"ndarray","array":{"type":"bytes","data":"9uaUZ+0KDMA="},"shape":[1],"dtype":"float64","order":"little"}],["coin_id",["celsius-degree-token"]],["predicted_clusters",[3]]]}}},"view":{"type":"object","name":"CDSView","id":"p4849","attributes":{"filter":{"type":"object","name":"AllIndices","id":"p4850"}}},"glyph":{"type":"object","name":"Scatter","id":"p4845","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"price_change_percentage_24h"},"y":{"type":"field","field":"price_change_percentage_7d"},"size":{"type":"value","value":5.477225575051661},"line_color":{"type":"value","value":"#6d904f"},"fill_color":{"type":"value","value":"#6d904f"},"hatch_color":{"type":"value","value":"#6d904f"}}},"selection_glyph":{"type":"object","name":"Scatter","id":"p4852","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"price_change_percentage_24h"},"y":{"type":"field","field":"price_change_percentage_7d"},"size":{"type":"value","value":5.477225575051661},"angle":{"type":"value","value":0.0},"line_color":{"type":"value","value":"#6d904f"},"line_alpha":{"type":"value","value":1.0},"line_width":{"type":"value","value":1},"line_join":{"type":"value","value":"bevel"},"line_cap":{"type":"value","value":"butt"},"line_dash":{"type":"value","value":[]},"line_dash_offset":{"type":"value","value":0},"fill_color":{"type":"value","value":"#6d904f"},"fill_alpha":{"type":"value","value":1.0},"hatch_color":{"type":"value","value":"#6d904f"},"hatch_alpha":{"type":"value","value":1.0},"hatch_scale":{"type":"value","value":12.0},"hatch_pattern":{"type":"value","value":null},"hatch_weight":{"type":"value","value":1.0},"marker":{"type":"value","value":"circle"}}},"nonselection_glyph":{"type":"object","name":"Scatter","id":"p4846","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"price_change_percentage_24h"},"y":{"type":"field","field":"price_change_percentage_7d"},"size":{"type":"value","value":5.477225575051661},"line_color":{"type":"value","value":"#6d904f"},"line_alpha":{"type":"value","value":0.1},"fill_color":{"type":"value","value":"#6d904f"},"fill_alpha":{"type":"value","value":0.1},"hatch_color":{"type":"value","value":"#6d904f"},"hatch_alpha":{"type":"value","value":0.1}}},"muted_glyph":{"type":"object","name":"Scatter","id":"p4847","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"price_change_percentage_24h"},"y":{"type":"field","field":"price_change_percentage_7d"},"size":{"type":"value","value":5.477225575051661},"line_color":{"type":"value","value":"#6d904f"},"line_alpha":{"type":"value","value":0.2},"fill_color":{"type":"value","value":"#6d904f"},"fill_alpha":{"type":"value","value":0.2},"hatch_color":{"type":"value","value":"#6d904f"},"hatch_alpha":{"type":"value","value":0.2}}}}}],"tooltips":[["predicted clusters","@{predicted_clusters}"],["price_change_percentage_24h","@{price_change_percentage_24h}"],["price_change_percentage_7d","@{price_change_percentage_7d}"],["coin_id","@{coin_id}"]]}},{"type":"object","name":"HoverTool","id":"p4857","attributes":{"tags":["hv_created"],"renderers":[{"id":"p4903"},{"id":"p4918"},{"id":"p4932"},{"id":"p4946"}],"tooltips":[["Predicted Clusters","@{Predicted_Clusters}"],["price_change_percentage_24h","@{price_change_percentage_24h}"],["price_change_percentage_7d","@{price_change_percentage_7d}"],["coin_id","@{coin_id}"]]}}]}},{"type":"object","name":"SaveTool","id":"p4969"},{"type":"object","name":"ToolProxy","id":"p4970","attributes":{"tools":[{"type":"object","name":"PanTool","id":"p4792"},{"type":"object","name":"PanTool","id":"p4890"}]}},{"type":"object","name":"ToolProxy","id":"p4971","attributes":{"tools":[{"type":"object","name":"BoxZoomTool","id":"p4793","attributes":{"overlay":{"type":"object","name":"BoxAnnotation","id":"p4794","attributes":{"syncable":false,"level":"overlay","visible":false,"left_units":"canvas","right_units":"canvas","bottom_units":"canvas","top_units":"canvas","line_color":"black","line_alpha":1.0,"line_width":2,"line_dash":[4,4],"fill_color":"lightgrey","fill_alpha":0.5}}}},{"type":"object","name":"BoxZoomTool","id":"p4891","attributes":{"overlay":{"type":"object","name":"BoxAnnotation","id":"p4892","attributes":{"syncable":false,"level":"overlay","visible":false,"left_units":"canvas","right_units":"canvas","bottom_units":"canvas","top_units":"canvas","line_color":"black","line_alpha":1.0,"line_width":2,"line_dash":[4,4],"fill_color":"lightgrey","fill_alpha":0.5}}}}]}},{"type":"object","name":"ToolProxy","id":"p4972","attributes":{"tools":[{"type":"object","name":"ResetTool","id":"p4795"},{"type":"object","name":"ResetTool","id":"p4893"}]}}]}},"children":[[{"type":"object","name":"Figure","id":"p4769","attributes":{"width":700,"height":300,"sizing_mode":"fixed","align":"start","x_range":{"type":"object","name":"Range1d","id":"p4753","attributes":{"tags":[[["price_change_percentage_24h","price_change_percentage_24h",null]],[]],"start":-5.526782965204708,"end":8.29865769927953,"reset_start":-5.526782965204708,"reset_end":8.29865769927953}},"y_range":{"type":"object","name":"Range1d","id":"p4754","attributes":{"tags":[[["price_change_percentage_7d","price_change_percentage_7d",null]],{"type":"map","entries":[["invert_yaxis",false],["autorange",false]]}],"start":-4.582209243538451,"end":8.340274168490136,"reset_start":-4.582209243538451,"reset_end":8.340274168490136}},"x_scale":{"type":"object","name":"LinearScale","id":"p4779"},"y_scale":{"type":"object","name":"LinearScale","id":"p4780"},"title":{"type":"object","name":"Title","id":"p4772","attributes":{"text":"Segment Clusters","text_color":"black","text_font_size":"12pt"}},"renderers":[{"id":"p4805"},{"id":"p4820"},{"id":"p4834"},{"id":"p4848"}],"toolbar":{"type":"object","name":"Toolbar","id":"p4778","attributes":{"tools":[{"id":"p4758"},{"id":"p4759"},{"type":"object","name":"SaveTool","id":"p4791"},{"id":"p4792"},{"id":"p4793"},{"id":"p4795"}],"active_drag":{"id":"p4792"},"active_scroll":{"id":"p4758"}}},"toolbar_location":null,"left":[{"type":"object","name":"LinearAxis","id":"p4786","attributes":{"ticker":{"type":"object","name":"BasicTicker","id":"p4787","attributes":{"mantissas":[1,2,5]}},"formatter":{"type":"object","name":"BasicTickFormatter","id":"p4788"},"axis_label":"Price Change Percentage 7 Days","major_label_policy":{"type":"object","name":"AllLabels","id":"p4789"}}}],"right":[{"type":"object","name":"Legend","id":"p4808","attributes":{"location":[0,0],"title":"predicted clusters","click_policy":"mute","items":[{"type":"object","name":"LegendItem","id":"p4809","attributes":{"label":{"type":"value","value":"0"},"renderers":[{"id":"p4805"}]}},{"type":"object","name":"LegendItem","id":"p4823","attributes":{"label":{"type":"value","value":"1"},"renderers":[{"id":"p4820"}]}},{"type":"object","name":"LegendItem","id":"p4837","attributes":{"label":{"type":"value","value":"2"},"renderers":[{"id":"p4834"}]}},{"type":"object","name":"LegendItem","id":"p4851","attributes":{"label":{"type":"value","value":"3"},"renderers":[{"id":"p4848"}]}}]}}],"below":[{"type":"object","name":"LinearAxis","id":"p4781","attributes":{"ticker":{"type":"object","name":"BasicTicker","id":"p4782","attributes":{"mantissas":[1,2,5]}},"formatter":{"type":"object","name":"BasicTickFormatter","id":"p4783"},"axis_label":"Price Change Percentage 24 Hours","major_label_policy":{"type":"object","name":"AllLabels","id":"p4784"}}}],"center":[{"type":"object","name":"Grid","id":"p4785","attributes":{"axis":{"id":"p4781"},"grid_line_color":null}},{"type":"object","name":"Grid","id":"p4790","attributes":{"dimension":1,"axis":{"id":"p4786"},"grid_line_color":null}}],"min_border_top":10,"min_border_bottom":10,"min_border_left":10,"min_border_right":10,"output_backend":"webgl"}},0,0],[{"type":"object","name":"Figure","id":"p4867","attributes":{"width":700,"height":300,"sizing_mode":"fixed","align":"start","x_range":{"id":"p4753"},"y_range":{"id":"p4754"},"x_scale":{"type":"object","name":"LinearScale","id":"p4877"},"y_scale":{"type":"object","name":"LinearScale","id":"p4878"},"title":{"type":"object","name":"Title","id":"p4870","attributes":{"text":"Segement Clusters","text_color":"black","text_font_size":"12pt"}},"renderers":[{"id":"p4903"},{"id":"p4918"},{"id":"p4932"},{"id":"p4946"}],"toolbar":{"type":"object","name":"Toolbar","id":"p4876","attributes":{"tools":[{"id":"p4856"},{"id":"p4857"},{"type":"object","name":"SaveTool","id":"p4889"},{"id":"p4890"},{"id":"p4891"},{"id":"p4893"}],"active_drag":{"id":"p4890"},"active_scroll":{"id":"p4856"}}},"toolbar_location":null,"left":[{"type":"object","name":"LinearAxis","id":"p4884","attributes":{"ticker":{"type":"object","name":"BasicTicker","id":"p4885","attributes":{"mantissas":[1,2,5]}},"formatter":{"type":"object","name":"BasicTickFormatter","id":"p4886"},"axis_label":"Price Change Percentage 7 Days","major_label_policy":{"type":"object","name":"AllLabels","id":"p4887"}}}],"right":[{"type":"object","name":"Legend","id":"p4906","attributes":{"location":[0,0],"title":"Predicted Clusters","click_policy":"mute","items":[{"type":"object","name":"LegendItem","id":"p4907","attributes":{"label":{"type":"value","value":"0"},"renderers":[{"id":"p4903"}]}},{"type":"object","name":"LegendItem","id":"p4921","attributes":{"label":{"type":"value","value":"1"},"renderers":[{"id":"p4918"}]}},{"type":"object","name":"LegendItem","id":"p4935","attributes":{"label":{"type":"value","value":"2"},"renderers":[{"id":"p4932"}]}},{"type":"object","name":"LegendItem","id":"p4949","attributes":{"label":{"type":"value","value":"3"},"renderers":[{"id":"p4946"}]}}]}}],"below":[{"type":"object","name":"LinearAxis","id":"p4879","attributes":{"ticker":{"type":"object","name":"BasicTicker","id":"p4880","attributes":{"mantissas":[1,2,5]}},"formatter":{"type":"object","name":"BasicTickFormatter","id":"p4881"},"axis_label":"Price Change Percentage 24 Hours","major_label_policy":{"type":"object","name":"AllLabels","id":"p4882"}}}],"center":[{"type":"object","name":"Grid","id":"p4883","attributes":{"axis":{"id":"p4879"},"grid_line_color":null}},{"type":"object","name":"Grid","id":"p4888","attributes":{"dimension":1,"axis":{"id":"p4884"},"grid_line_color":null}}],"min_border_top":10,"min_border_bottom":10,"min_border_left":10,"min_border_right":10,"output_backend":"webgl"}},0,1]]}},{"type":"object","name":"Spacer","id":"p4975","attributes":{"name":"HSpacer10121","stylesheets":["\n:host(.pn-loading.pn-arc):before, .pn-loading.pn-arc:before {\n  background-image: url(\"data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHN0eWxlPSJtYXJnaW46IGF1dG87IGJhY2tncm91bmQ6IG5vbmU7IGRpc3BsYXk6IGJsb2NrOyBzaGFwZS1yZW5kZXJpbmc6IGF1dG87IiB2aWV3Qm94PSIwIDAgMTAwIDEwMCIgcHJlc2VydmVBc3BlY3RSYXRpbz0ieE1pZFlNaWQiPiAgPGNpcmNsZSBjeD0iNTAiIGN5PSI1MCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjYzNjM2MzIiBzdHJva2Utd2lkdGg9IjEwIiByPSIzNSIgc3Ryb2tlLWRhc2hhcnJheT0iMTY0LjkzMzYxNDMxMzQ2NDE1IDU2Ljk3Nzg3MTQzNzgyMTM4Ij4gICAgPGFuaW1hdGVUcmFuc2Zvcm0gYXR0cmlidXRlTmFtZT0idHJhbnNmb3JtIiB0eXBlPSJyb3RhdGUiIHJlcGVhdENvdW50PSJpbmRlZmluaXRlIiBkdXI9IjFzIiB2YWx1ZXM9IjAgNTAgNTA7MzYwIDUwIDUwIiBrZXlUaW1lcz0iMDsxIj48L2FuaW1hdGVUcmFuc2Zvcm0+ICA8L2NpcmNsZT48L3N2Zz4=\");\n  background-size: auto calc(min(50%, 400px));\n}",{"id":"p4751"},{"id":"p4749"},{"id":"p4750"}],"margin":0,"sizing_mode":"stretch_width","align":"start"}}]}}],"defs":[{"type":"model","name":"ReactiveHTML1"},{"type":"model","name":"FlexBox1","properties":[{"name":"align_content","kind":"Any","default":"flex-start"},{"name":"align_items","kind":"Any","default":"flex-start"},{"name":"flex_direction","kind":"Any","default":"row"},{"name":"flex_wrap","kind":"Any","default":"wrap"},{"name":"justify_content","kind":"Any","default":"flex-start"}]},{"type":"model","name":"FloatPanel1","properties":[{"name":"config","kind":"Any","default":{"type":"map"}},{"name":"contained","kind":"Any","default":true},{"name":"position","kind":"Any","default":"right-top"},{"name":"offsetx","kind":"Any","default":null},{"name":"offsety","kind":"Any","default":null},{"name":"theme","kind":"Any","default":"primary"},{"name":"status","kind":"Any","default":"normalized"}]},{"type":"model","name":"GridStack1","properties":[{"name":"mode","kind":"Any","default":"warn"},{"name":"ncols","kind":"Any","default":null},{"name":"nrows","kind":"Any","default":null},{"name":"allow_resize","kind":"Any","default":true},{"name":"allow_drag","kind":"Any","default":true},{"name":"state","kind":"Any","default":[]}]},{"type":"model","name":"drag1","properties":[{"name":"slider_width","kind":"Any","default":5},{"name":"slider_color","kind":"Any","default":"black"},{"name":"value","kind":"Any","default":50}]},{"type":"model","name":"click1","properties":[{"name":"terminal_output","kind":"Any","default":""},{"name":"debug_name","kind":"Any","default":""},{"name":"clears","kind":"Any","default":0}]},{"type":"model","name":"FastWrapper1","properties":[{"name":"object","kind":"Any","default":null},{"name":"style","kind":"Any","default":null}]},{"type":"model","name":"NotificationAreaBase1","properties":[{"name":"js_events","kind":"Any","default":{"type":"map"}},{"name":"position","kind":"Any","default":"bottom-right"},{"name":"_clear","kind":"Any","default":0}]},{"type":"model","name":"NotificationArea1","properties":[{"name":"js_events","kind":"Any","default":{"type":"map"}},{"name":"notifications","kind":"Any","default":[]},{"name":"position","kind":"Any","default":"bottom-right"},{"name":"_clear","kind":"Any","default":0},{"name":"types","kind":"Any","default":[{"type":"map","entries":[["type","warning"],["background","#ffc107"],["icon",{"type":"map","entries":[["className","fas fa-exclamation-triangle"],["tagName","i"],["color","white"]]}]]},{"type":"map","entries":[["type","info"],["background","#007bff"],["icon",{"type":"map","entries":[["className","fas fa-info-circle"],["tagName","i"],["color","white"]]}]]}]}]},{"type":"model","name":"Notification","properties":[{"name":"background","kind":"Any","default":null},{"name":"duration","kind":"Any","default":3000},{"name":"icon","kind":"Any","default":null},{"name":"message","kind":"Any","default":""},{"name":"notification_type","kind":"Any","default":null},{"name":"_destroyed","kind":"Any","default":false}]},{"type":"model","name":"TemplateActions1","properties":[{"name":"open_modal","kind":"Any","default":0},{"name":"close_modal","kind":"Any","default":0}]},{"type":"model","name":"BootstrapTemplateActions1","properties":[{"name":"open_modal","kind":"Any","default":0},{"name":"close_modal","kind":"Any","default":0}]},{"type":"model","name":"MaterialTemplateActions1","properties":[{"name":"open_modal","kind":"Any","default":0},{"name":"close_modal","kind":"Any","default":0}]}]}};
  var render_items = [{"docid":"479ec3f4-1c8e-4a97-86bf-0fcce7323cce","roots":{"p4748":"f7ae49d1-5411-4876-99f9-bcee215b4502"},"root_ids":["p4748"]}];
  var docs = Object.values(docs_json)
  if (!docs) {
    return
  }
  const py_version = docs[0].version.replace('rc', '-rc.').replace('.dev', '-dev.')
  const is_dev = py_version.indexOf("+") !== -1 || py_version.indexOf("-") !== -1
  function embed_document(root) {
    var Bokeh = get_bokeh(root)
    Bokeh.embed.embed_items_notebook(docs_json, render_items);
    for (const render_item of render_items) {
      for (const root_id of render_item.root_ids) {
	const id_el = document.getElementById(root_id)
	if (id_el.children.length && (id_el.children[0].className === 'bk-root')) {
	  const root_el = id_el.children[0]
	  root_el.id = root_el.id + '-rendered'
	}
      }
    }
  }
  function get_bokeh(root) {
    if (root.Bokeh === undefined) {
      return null
    } else if (root.Bokeh.version !== py_version && !is_dev) {
      if (root.Bokeh.versions === undefined || !root.Bokeh.versions.has(py_version)) {
	return null
      }
      return root.Bokeh.versions.get(py_version);
    } else if (root.Bokeh.version === py_version) {
      return root.Bokeh
    }
    return null
  }
  function is_loaded(root) {
    var Bokeh = get_bokeh(root)
    return (Bokeh != null && Bokeh.Panel !== undefined)
  }
  if (is_loaded(root)) {
    embed_document(root);
  } else {
    var attempts = 0;
    var timer = setInterval(function(root) {
      if (is_loaded(root)) {
        clearInterval(timer);
        embed_document(root);
      } else if (document.readyState == "complete") {
        attempts++;
        if (attempts > 200) {
          clearInterval(timer);
	  var Bokeh = get_bokeh(root)
	  if (Bokeh == null || Bokeh.Panel == null) {
            console.warn("Panel: ERROR: Unable to run Panel code because Bokeh or Panel library is missing");
	  } else {
	    console.warn("Panel: WARNING: Attempting to render but not all required libraries could be resolved.")
	    embed_document(root)
	  }
        }
      }
    }, 25, root)
  }
})(window);</script>



#### Answer the following question: After visually analyzing the cluster analysis results, what is the impact of using fewer features to cluster the data using K-Means?

  * **Question:** After visually analyzing the cluster analysis results, what is the impact of using fewer features to cluster the data using K-Means?

  * **Answer:** The inertia drop is much steeper when using fewer features, and the predicted clusters are more spaced out and few overlaps. 


```python

```
