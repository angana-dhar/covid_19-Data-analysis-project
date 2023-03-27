#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime


# In[61]:


covid_df =pd.read_csv("C:/Users/dell/OneDrive - vitbhopal.ac.in/Documents/covid_19_india.csv")


# In[19]:


covid_df.head(10)


# In[20]:


covid_df.info()


# In[21]:


covid_df.describe()


# In[23]:


vaccine_df =pd.read_csv("C:/Users/dell/OneDrive - vitbhopal.ac.in/Documents/covid_vaccine_statewise.csv")


# In[24]:


vaccine_df.head(20)


# In[25]:


vaccine_df.info()


# In[26]:


covid_df=pd.read_csv("C:/Users/dell/OneDrive - vitbhopal.ac.in/Documents/covid_19_india.csv")


# In[27]:


covid_df.head(10)


# In[28]:


covid_df.drop(["Sno","Time","ConfirmedIndianNational","ConfirmedForeignNational"],inplace=True,axis=1)


# In[29]:


covid_df.head(12)


# In[30]:


#changing the format of date 

covid_df['Date'] = pd.to_datetime(covid_df['Date'], format = '%Y-%m-%d')


# In[31]:


covid_df.head(2)


# In[32]:


#finding total number of active cases
#total no. of confirmed cases - (total no. of cured+total no. of deaths)
covid_df['Active_Cases'] = covid_df['Confirmed']-(covid_df['Cured']+covid_df['Deaths'])
covid_df.tail()


# In[33]:


#creating a pibot table(adding confirmed cases and cured cases every states and union territory)
#using pibot_table function
statewise = pd.pivot_table(covid_df, values=["Confirmed","Deaths","Cured"],index = "State/UnionTerritory",aggfunc=max)


# In[49]:


#finding recovery rate 
#(total number of cured cases divided by total number of confiremed cases)*100
statewise["Recovery Rate"] = (statewise["Cured"]*100/statewise["Confirmed"])


# In[34]:


statewise[statewise.index.duplicated()]


# In[35]:


#mortality rate 
#(total number of deaths/total number of confirmed cases)*100
statewise["Mortality Rate"] = (statewise["Deaths"]*100/statewise["Confirmed"])


# In[36]:


#sort the values based on the confirmed cases in ascending  order
statewise = statewise.sort_values(by="Confirmed",ascending=True)


# In[37]:


#now we are going to plot pivot table using a nice visual for that we will use mybackground_gradient function inside that function we will insert cmap parameter
statewise.style.background_gradient(cmap="cubehelix")


# In[38]:


#top t10 active cases states
top_10_active_cases = covid_df.groupby(by='State/UnionTerritory').max()[['Active_Cases','Date']].sort_values(by=['Active_Cases'],ascending=False).reset_index()


# In[39]:


fig = plt.figure(figsize=(16,9))


# In[40]:


plt.title("TOP 10 STATES WITH MOST ACTIVE CASES IN INDIA",size=25)


# In[41]:


ax = sns.barplot(data = top_10_active_cases.iloc[:10],y="Active_Cases",x = "State/UnionTerritory",linewidth=2,edgecolor='purple')


# In[42]:


top_10_active_cases = covid_df.groupby(by='State/UnionTerritory').max()[['Active_Cases','Date']].sort_values(by=['Active_Cases'],ascending=False).reset_index()
fig = plt.figure(figsize=(16,9))
plt.title("TOP 10 STATES WITH MOST ACTIVE CASES IN INDIA",size=25)    
ax = sns.barplot(data = top_10_active_cases.iloc[:10],y="Active_Cases",x = "State/UnionTerritory",linewidth=2,edgecolor='purple')
plt.xlabel("states")
plt.ylabel("Total Active Cases")
plt.show()


# In[73]:


#top 10 states based on total number of death reports
top_10_deaths = covid_df.groupby(by='State/UnionTerritory').max()[['Deaths','Date']].sort_values(by=['Deaths'],ascending=False).reset_index()
fig = plt.figure(figsize=(16,9))
plt.title("TOP 10 STATES WITH MAXIMUM NUMBER OF DEATHS",size=25)    
ax = sns.barplot(data = top_10_deaths.iloc[:12],y="Deaths",x = "State/UnionTerritory",linewidth=2,edgecolor='Green')
plt.xlabel("states")
plt.ylabel("Total Number of Deaths")
plt.show()


# In[55]:


#Growth trend
fig = plt.figure(figsize = (12,6))
ax = sns.lineplot(data = covid_df[covid_df['State/UnionTerritory'].isin(['Maharashtra','Karnataka',
                                                                         'Kerala','Tamil Nadu','Uttar Pradesh']),x == 'Date', y == 'Active_Cases', hue == 'State/UnionTerritory'])
ax.set_title("Top 5 Affected States in India",size==16)


# In[56]:


vaccine_df.head()


# In[64]:


vaccine_df.rename(columns ={'Updated On':'vaccine_date'},inplace = True)


# In[65]:


vaccine_df.head(10)


# In[67]:


vaccine_df.info()


# In[69]:


vaccine_df.isnull().sum()


# In[75]:


vaccination = vaccine_df.drop(columns = ['Sputnik V (Doses Administered)','AEFI','18-44 Years (Doses Administered)','45-60 Years (Doses Administered)','60+ Years (Doses Administered)'],axis=1)


# In[76]:


vaccination.head()


# In[78]:


#male vs female vaccination 
male = vaccination["Male(Individuals Vaccinated)"].sum()
female = vaccination["Female(Individuals Vaccinated)"].sum()
px.pie(names=["Male","Female"],values = [male,female],title="Male and Female vaccination")


# In[80]:


#removing rows where state is named as India
vaccine = vaccine_df[vaccine_df.State!='India']
vaccine


# In[81]:


vaccine.rename(columns ={'Total Individuals Vaccinated':'Total'},inplace = True)


# In[82]:


vaccine.head(4)


# In[83]:


#finding the states with most number of vaccinated induviduals and the state with the least no.s of vaccinated individuals
#most vaccinated state 
max_vac = vaccine.groupby('State')['Total'].sum().to_frame('Total')
max_vac = max_vac.sort_values('Total',ascending=False)[:5]
max_vac


# In[88]:


fig = plt.figure(figsize =(10,5))
plt.title("Top 5 Vaccinated States in India",size = 20)
x = sns.barplot(data = max_vac.iloc[:10],y=max_vac.Total,x=max_vac.index,linewidth=2,edgecolor='red')
plt.xlabel("States")
plt.ylabel("Vaccination")
plt.show()


# In[91]:


#least vaccinated state 
min_vac = vaccine.groupby('State')['Total'].sum().to_frame('Total')
min_vac = min_vac.sort_values('Total',ascending=True)[:5]
min_vac


# In[92]:


fig = plt.figure(figsize =(10,5))
plt.title("bottom 5 Vaccinated States in India",size = 20)
x = sns.barplot(data =min_vac.iloc[:10],y=min_vac.Total,x=min_vac.index,linewidth=2,edgecolor='red')
plt.xlabel("States")
plt.ylabel("Vaccination")
plt.show()

