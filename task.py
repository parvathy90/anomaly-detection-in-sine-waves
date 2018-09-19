#problem 1
import csv
import pandas as pd
from datetime import date, datetime, timedelta
#read data from xlsx format
data_xls = pd.read_excel('data_extension.xlsx', index_col=None)
#data_xls.to_csv('your_csv.csv', encoding='utf-8')
list1=[]
for i in range(len(data_xls)):
	list1.append(1)
#creating new column with value
data_xls["Parent ID"] = list1
#saving the data to a csv file
data_xls.to_csv('parent_ids.csv')

#problem 2
data_xls1 = pd.read_excel('data_date.xlsx', index_col=None)
orderid=data_xls1['Order ID'].values.tolist()
quantity=data_xls1['Quantity'].values.tolist()
archive=data_xls1['Archive'].values.tolist()
city=data_xls1['City'].values.tolist()
#taking start and end date
p=(data_xls1.iloc[:,3])
pe=(data_xls1.iloc[:,4])
#converting to datetime format
p1= pd.to_datetime(p, format="%Y/%m/%d")
p1e= pd.to_datetime(pe, format="%Y/%m/%d")
start_d=p1
end_d=p1e
def perdelta(start, end, delta):
    curr = start
    while curr < end:
        yield curr
        curr += delta
list2=[]
split_start=[]
split_end=[]
start_date=[]
end_date=[]
quant=[]
archiv=[]
order=[]
cit=[]
#splitting the start and end dates into 30 days and appending the other columns
for item,item1,i,j,k,l,m,n in zip(p1,p1e,orderid,quantity,start_d,end_d,archive,city):
	for result in perdelta(item, item1, timedelta(days=31)):
		if(item1!=result):
			split_start.append(result)
			order.append(i)
			quant.append(j)
			start_date.append(k)
			end_date.append(l)
			archiv.append(m)
			cit.append(n)
		end_time=result-timedelta(days=1)	
		if(end_time>item):
			split_end.append(end_time)
		if((item1-result).days<=31):
			split_end.append(item1)	
	if(item==item1):
		split_end.append(item1)
		split_start.append(item)
		order.append(i)
		quant.append(j)
		start_date.append(k)
		end_date.append(l)
		archiv.append(m)		
		cit.append(n)
sp_e=[]
sp_s=[]
s_d=[]
e_d=[]
list_1=order
#converting datetime into actual format
for i in range(len(split_start)):	
	l=(split_end[i].strftime('%m/%d/%Y'))
	l1=(split_start[i].strftime('%m/%d/%Y'))
	l2=(start_date[i].strftime('%m/%d/%Y'))
	l3=(end_date[i].strftime('%m/%d/%Y'))
	sp_e.append(l)
	sp_s.append(l1)
	s_d.append(l2)
	e_d.append(l3)
#encode different formats into strings and integers
csv_data=[]
order=[x.encode('UTF8') for x in order]
archiv=[int(x) for x in archiv]
quant=[int(x) for x in quant]
sp_s=[x.encode('UTF8') for x in sp_s]
sp_e=[x.encode('UTF8') for x in sp_e]
s_d=[x.encode('UTF8') for x in s_d]
e_d=[x.encode('UTF8') for x in e_d]
cit=[x.encode('UTF8') for x in cit]
#creating 8 dimensional list using all columns
csv_data = [list(a) for a in zip(order, archiv,quant,s_d,e_d,sp_s,sp_e,cit)]
#converting list into dataframe and storing it as csv file
my_df = pd.DataFrame(csv_data)
my_df.to_csv('split_dates.csv', index=False, header=False)


