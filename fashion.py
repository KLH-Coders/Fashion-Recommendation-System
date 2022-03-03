
df_pyspark.show()
df_pyspark.printSchema()
df_pyspark.head(3)
df_pyspark.select(['id','gender']).show()
from pyspark.sql.functions import concat, col, lit
df_pyspark = df_pyspark.withColumn('image',concat(df_pyspark['id'],lit('.jpg')))
df_pyspark.show()
import pandas as pd
spark.conf.set("spark.sql.execution.arrow.enabled", "true")
df = df_pyspark.toPandas()
import cv2
def plot_figures(figures, nrows = 1, ncols=1,figsize=(8, 8)):

    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows,figsize=figsize)
    for ind,title in enumerate(figures):
        axeslist.ravel()[ind].imshow(cv2.cvtColor(figures[title], cv2.COLOR_BGR2RGB))
        axeslist.ravel()[ind].set_title(title)
        axeslist.ravel()[ind].set_axis_off()
    plt.tight_layout() # optional
    
def img_path(img):
    return path+"/images/"+img

def load_image(img, resized_fac = 0.1):
    img     = cv2.imread(img_path(img))
    
    w, h, _ = img.shape
    resized = cv2.resize(img, (int(h*resized_fac), int(w*resized_fac)), interpolation = cv2.INTER_AREA)
    return resized
  import matplotlib.pyplot as plt
import numpy as np

figures = {'im'+str(i): load_image(row.image) for i, row in df.sample(6).iterrows()}
plot_figures(figures, 2, 3)
