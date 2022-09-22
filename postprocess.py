import streamlit as st
from utils import *
import csv
import datetime
import pandas as pd
import numpy as np
import math
import os

# Color Palette
CP_R = '\033[31m'
CP_G = '\033[32m'
CP_Y = '\033[33m'
CP_C = '\033[36m'
CP_0 = '\033[0m'

deviceType = 'FPGA'

class PostProcess:
	"""
	Create dataframes, display and save the models
	"""
		
	def __init__(self):
		self.header_sum = ['Hardware', 'numDLA', 'numClustersPerDLA', 
						'AIapp', 'AImodel', 'Dataset', 'BatchSize', 'AvgLatencyRaw(ms)', 
						'AvgLatencyDLA(ms)', 'MeanAccuracy(%)', 'AvgIPSraw', 
						'AvgIPSdla']
		self.df_sum = pd.DataFrame(columns=self.header_sum)

		self.header = ['Hardware', 'InputSample', 'numDLA', 'numClustersPerDLA', 
					'AIapp', 'AImodel', 'Dataset', 'BatchSize', 'LatencyRaw(ms)', 'LatencyDLA(ms)', 'Accuracy', 'PredictedLabel', 
					'AccuracyTop5', 'PredictedLabelsTop5', 'IPSraw', 'IPSdla']

		self.df = pd.DataFrame(columns=self.header)

		ct = datetime.datetime.now()
		fname = str(ct).split(' ')[0]+'_'+str(ct).split(' ')[1]+'.csv'

		self.log_name_sum = 'SummaryLog_' + fname
		self.log_name = 'RawLog_' + fname


	def save(self, df=None):
		if df is None:
			df = self.df
		df.to_csv('logs/' + self.log_name, index=False)


		#st.dataframe(df)

		""" with open('logs/logs_each.csv', 'w') as f:
			writer = csv.writer(f)                                        # create the csv writer

			writer.writerow(self.header)
			writer.writerow(self.metrics) """

	def save_sum(self, df=None):
		if df is None:
			df = self.df_sum
		df.to_csv('logs/' + self.log_name_sum, index=False)
		#st.dataframe(df)

		""" with open('logs/logs.csv', 'w') as f:
			writer = csv.writer(f)                                        # create the csv writer

			writer.writerow(self.header_sum)
			writer.writerow(self.metrics_sum) """


	def postProcess(self, modelName, inp, nDLA, nClusters, appChoice, modelChoice, dataChoice, raw_latency, 
					dla_latency, batch, acc, acc5, pred_label1, pred_label5, thrpt_raw, thrpt_dla):

		# Save individual results to csv
		self.metrics = [deviceType, inp, nDLA, nClusters, appChoice, modelChoice, dataChoice, batch, 
									raw_latency, dla_latency, acc, pred_label1, acc5, pred_label5, thrpt_raw, 
									thrpt_dla]

		self.df.loc[len(self.df.index)] = self.metrics



	def postProcessSum(self):

		hardware = self.df.iloc[0]['Hardware']
		model = self.df.iloc[0]['AImodel']
		batchSize = self.df.iloc[0]['BatchSize']
		numDLA = self.df.iloc[0]['numDLA']
		numClustersPerDLA = self.df.iloc[0]['numClustersPerDLA']
		appChoice = self.df.iloc[0]['AIapp']
		dataChoice = self.df.iloc[0]['Dataset']

		avg_raw_lat = self.df['LatencyRaw(ms)'].mean().round(1)
		avg_dla_lat = self.df['LatencyDLA(ms)'].mean().round(1)
		mean_acc = self.df['Accuracy'].mean().round(2)
		avg_thrpt_raw = round(1000/avg_raw_lat,2).round(1)
		avg_thrpt_dla = round(1000/avg_dla_lat,2).round(1)

		col1, col2, col3 = st.columns([2,2,10])

		col1.subheader("Summary")
		col2.subheader("...")
		col1.write(model+" on "+hardware)
		col2.write("Batch Size: "+str(batchSize))
		col1.write("#DLA: "+str(numDLA))
		col2.write("#Clusters: "+str(numClustersPerDLA))
		col1.metric(label="Avg Raw Latency (ms)", value=avg_raw_lat)
		col2.metric(label="Avg DLA Latency (ms)", value=avg_dla_lat)
		col1.metric(label="Avg Raw Inferences per Second", value=avg_thrpt_raw)
		col2.metric(label="Avg DLA Inferences per Second", value=avg_thrpt_dla)
		col1.metric(label="Top-1 Accuracy (%)", value=mean_acc)

		pic_df = self.df.iloc[0:5]
		st.dataframe(pic_df)

		self.df = self.df.drop(['Hardware', 'InputSample','PredictedLabel','AccuracyTop5','PredictedLabelsTop5'], axis=1)
		col3.subheader('Individual inferences')
		col3.table(self.df.head(10))


		pic_list = pic_df['InputSample'].values.tolist()
		pred_list = pic_df['PredictedLabelsTop5'].values.tolist()
		acc_list = pic_df['AccuracyTop5'].values.tolist()

		cwd_path = os.path.abspath(os.getcwd())
		imgpath = cwd_path + '/Datasets/imagenet1k-test/dataset/'

		col1, col2 = st.columns([2,8])
		col1.header('Sample Image')
		col2.header('Top-5 Prediction Labels')
		
		for idx, image in enumerate(pic_list):
			col1, col2 = st.columns([2,8])
			file = imgpath + image
			col1.image(file, width=200)
			#col2.write(pred_list[idx])
			col2.write(dict(zip(pred_list[idx], acc_list[idx])))
			
		# Save mean results to csv
		self.metrics_sum = [hardware, numDLA, numClustersPerDLA, appChoice, model, dataChoice, batchSize, 
								avg_raw_lat, avg_dla_lat, mean_acc, avg_thrpt_raw, avg_thrpt_dla]

		self.df_sum.loc[len(self.df_sum.index)] = self.metrics_sum
		self.save_sum(self.df_sum)
	
	def postProcessDetSum(self):
		hardware = self.df.iloc[0]['Hardware']
		model = self.df.iloc[0]['AImodel']
		batchSize = self.df.iloc[0]['BatchSize']
		numDLA = self.df.iloc[0]['numDLA']
		numClustersPerDLA = self.df.iloc[0]['numClustersPerDLA']
		appChoice = self.df.iloc[0]['AIapp']
		dataChoice = self.df.iloc[0]['Dataset']

		avg_raw_lat = self.df['LatencyRaw(ms)'].mean().round(1)
		avg_dla_lat = self.df['LatencyDLA(ms)'].mean().round(1)
		#mean_acc = self.df['Accuracy'].mean().round(2)
		mean_acc = 0.0
		avg_thrpt_raw = round(1000/avg_raw_lat,2).round(1)
		avg_thrpt_dla = round(1000/avg_dla_lat,2).round(1)

		col1, col2, col3 = st.columns([2,2,10])

		col1.subheader("Summary")
		col2.subheader("...")
		col1.write(model+" on "+hardware)
		col2.write("Batch Size: "+str(batchSize))
		col1.write("#DLA: "+str(numDLA))
		col2.write("#Clusters: "+str(numClustersPerDLA))
		col1.metric(label="Avg Raw Latency (ms)", value=avg_raw_lat)
		col2.metric(label="Avg DLA Latency (ms)", value=avg_dla_lat)
		col1.metric(label="Avg Raw Inferences per Second", value=avg_thrpt_raw)
		col2.metric(label="Avg DLA Inferences per Second", value=avg_thrpt_dla)
		#col1.metric(label="Top-1 Accuracy (%)", value=mean_acc)

		pic_df = self.df.iloc[0:5]
		#st.dataframe(pic_df)

		self.df = self.df.drop(['Hardware', 'InputSample','PredictedLabel','AccuracyTop5','PredictedLabelsTop5'], axis=1)
		col3.subheader('Individual inferences')
		col3.table(self.df.head(10))


		pic_list = pic_df['InputSample'].values.tolist()
		pred_list = pic_df['PredictedLabel'].values.tolist()

		cwd_path = (os.path.abspath(os.getcwd()))
		imgpath = cwd_path + '/testOuts/'

		col1, col2 = st.columns([2,8])
		col1.header('Sample Image')
		col2.header('Prediction Label')
		
		for idx, image in enumerate(pic_list):
			col1, col2 = st.columns([4,4])
			file = imgpath + image
			col1.image(file, width=400)
			col2.write(pred_list[idx])
			
		# Save mean results to csv
		self.metrics_sum = [hardware, numDLA, numClustersPerDLA, appChoice, model, dataChoice, batchSize, 
								avg_raw_lat, avg_dla_lat, mean_acc, avg_thrpt_raw, avg_thrpt_dla]

		self.df_sum.loc[len(self.df_sum.index)] = self.metrics_sum
		self.save_sum(self.df_sum)


