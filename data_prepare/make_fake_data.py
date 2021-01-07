# -*- coding:utf-8 -*-

# @Time    : 2021/1/7 下午3:30
# @Author  : ccj
# @Email   : 354840621@qq.com
# @File    : make_fake_data.py
# @Software: PyCharm

import numpy
import random
import pandas as pd
random.seed(100)
#1、生成数据的数量级

n = 100

#2、客户简介信息

#城市类型分为一线城市，二线城市，其他城市。 标注为1，2，3
city_type = []
for i in range(n):
    city_type_number = random.randint(1,3)
    if city_type_number == 1:
        city_type.append("一线城市")
    elif city_type_number == 2:
        city_type.append("二线城市")
    else:
        city_type.append("其他城市")

#信用分为良好，较差。 标注为1，2
credit = []
for i in range(n):
    credit_number = random.randint(1,2)
    if credit_number == 1:
        credit.append("良好")
    elif credit_number == 2:
        credit.append("较差")

#加入月份从1到12
join_month = []
for i in range(n):
    join_month_number = random.randint(1,12)
    join_month.append(join_month_number)


#国际漫游
gat_roaming_tag = []
for i in range(n):
    gat_roaming_tag_number = random.randint(0,1)
    gat_roaming_tag.append(gat_roaming_tag_number)

#省际漫游
provincial_roaming_tag = []
for i in range(n):
    provincial_roaming_tag_number = random.randint(0,1)
    provincial_roaming_tag.append(provincial_roaming_tag_number)

#两低用户标签
two_low_user_tag = []
for i in range(n):
    two_low_user_tag_number = random.randint(0,1)
    two_low_user_tag.append(two_low_user_tag_number)

#三低用户标签
three_low_user_tag = []
for i in range(n):
    three_low_user_tag_number = random.randint(0, 1)
    three_low_user_tag.append(two_low_user_tag_number)

#移动类型
mobile_type = []
for i in range(n):
    mobile_type_number = random.randint(0, 1)
    if mobile_type_number == 0:
        mobile_type.append('苹果手机')
    else:
        mobile_type.append('安卓手机')

# TDLTE
TDLTE_tag = []
for i in range(n):
    TDLTE_tag_nubmer = random.randint(0,1)
    TDLTE_tag.append(TDLTE_tag_nubmer)

# FDLTE
FDLTE_tag = []
for i in range(n):
    FDLTE_tag_nubmer = random.randint(0,1)
    FDLTE_tag.append(FDLTE_tag_nubmer)

customer_profiles_flatten = [city_type,credit,join_month,gat_roaming_tag,provincial_roaming_tag,two_low_user_tag,three_low_user_tag,mobile_type,TDLTE_tag,FDLTE_tag]

customer_profiles = []

for i in range(n):
    customer_profiles_i = []
    for j in range(len(customer_profiles_flatten)):
        customer_profiles_i.append(customer_profiles_flatten[j][i])
    customer_profiles.append(customer_profiles_i)

#2、订单细节
bill_details = []

#充值金额,分为50，100，150，200
recharge_amount  = []
for i in range(n):
    recharge_amount_number = random.randint(1,4)
    recharge_amount.append(recharge_amount_number*50)

#月费
monthly_fee = []
for i in range(n):
    monthly_fee_number = random.randint(38,1000)
    monthly_fee.append(monthly_fee_number)

#补助金额
grant_amount = []
for i in range(n):
    grant_amount_number = random.randint(0,38)
    grant_amount.append(grant_amount_number)

#欠款金额
arrears_amount = []
for i in range(n):
    arrears_amount_number = random.randint(0,50)
    arrears_amount.append(arrears_amount_number)

#套餐外语音费用
over_product_voice_income = []
for i in range(n):
    over_product_voice_income_number = random.randint(0,30)
    over_product_voice_income.append(over_product_voice_income_number)

#套餐外流量费用
over_product_stream_income = []
for i in range(n):
    over_product_stream_income_number = random.randint(0,48)
    over_product_stream_income.append(over_product_stream_income_number)


bill_details_flatten = [recharge_amount,monthly_fee,grant_amount,arrears_amount,over_product_stream_income,over_product_voice_income]

bill_details = []

for i in range(n):
    bill_details_i = []
    for j in range(len(bill_details_flatten)):
        bill_details_i.append(bill_details_flatten[j][i])
    bill_details.append(bill_details_i)

#流失状态, 1流失， 0未流失
churn_state_at_the_start_of_month = []
churn_state_at_the_end_of_month = []

for i in range(n):
    churn_state_at_the_start_of_month.append(0)
    if i< int(n*0.93):
        churn_state_at_the_end_of_month.append(0)
    else:
        churn_state_at_the_end_of_month.append(1)


#通话详情
call_details  = []

#漫游时长
roaming_call_duration = []
for i in range(n):
    roaming_call_duration_number = random.randint(0,3000)
    roaming_call_duration.append(roaming_call_duration_number)

#付费通话时长
paid_call_duration = []
for i in range(n):
    paid_call_duration_number = random.randint(0,289)
    paid_call_duration.append(paid_call_duration_number)

#是否套餐外有电话, 0表示没有， 1表示有
over_product_voice_tag = []
for i in range(n):
    over_product_voice_tag_number = random.randint(0,1)
    over_product_voice_tag.append(over_product_voice_tag_number)

#国内长途通话时长
domestic_Long_Distance_Call_Duration = []
for i in range(n):
    domestic_Long_Distance_Call_Duration_number = random.randint(0,298)
    domestic_Long_Distance_Call_Duration.append(domestic_Long_Distance_Call_Duration_number)

#国际长途电话时长
gat_International_Long_Distance_Call_Duration = []
for i in range(n):
    gat_International_Long_Distance_Call_Duration_number = random.randint(0,345)
    gat_International_Long_Distance_Call_Duration.append(gat_International_Long_Distance_Call_Duration_number)

#呼入个数
numbering_of_incoming_calls = []
for i in range(n):
    numbering_of_incoming_calls.append(random.randint(20,300))

#呼出个数
numbering_of_outgoing_calls = []
for i in range(n):
    numbering_of_outgoing_calls.append(random.randint(30,300))

call_details_flatten = [roaming_call_duration,paid_call_duration,over_product_voice_tag,domestic_Long_Distance_Call_Duration,gat_International_Long_Distance_Call_Duration,numbering_of_incoming_calls,numbering_of_outgoing_calls]

for i in range(n):
    call_details_i = []
    for j in range(len(call_details_flatten)):
        call_details_i.append(call_details_flatten[j][i])
    call_details.append(call_details_i)

#数据流量信息

# 付费数据流量
paid_Data_Traffic = []

for i in range(n):
    paid_Data_Traffic.append(random.randint(0,135))

#免费流量信息

free_Data_Traffic = []
for i in range(n):
    free_Data_Traffic.append(random.randint(0,145))

#省际流量
provincial_Data_Traffic = []
for i in range(n):
    provincial_Data_Traffic.append(random.randint(3,600))

#国内流量信息
domestic_Data_Traffic = []
for i in range(n):
    domestic_Data_Traffic.append(random.randint(0,10000))

#国际流量信息
international_Data_Traffic = []
for i in range(n):
    international_Data_Traffic.append(random.randint(30,3000))

#每天使用流量信息
data_Traffic_Used_Days = []
for i in range(n):
    data_Traffic_Used_Days.append(random.randint(10,100))

data_traffic_details = []

data_traffic_details_flatten = [paid_Data_Traffic,free_Data_Traffic,provincial_Data_Traffic,domestic_Data_Traffic,international_Data_Traffic,data_Traffic_Used_Days]
for i in range(n):
    data_traffic_details_i = []
    for j in range(len(data_traffic_details_flatten)):
        data_traffic_details_i.append(data_traffic_details_flatten[j][i])
    data_traffic_details.append(data_traffic_details_i)

#其他信息

#关机天数
shutdown_Days = []
for i in range(n):
    shutdown_Days.append(random.randint(0,30))

#短信条数
sms_numbers = []
for i in range(n):
    sms_numbers.append(random.randint(0,1000))

#促销标签
promotion_Tag = []

for i in range(n):
    promotion_Tag.append(random.randint(0,1))

other_information  = []

other_information_flatten = [shutdown_Days,sms_numbers,promotion_Tag]

for i in range(n):
    other_information_i = []
    for j in range(len(other_information_flatten)):
        other_information_i.append(other_information_flatten[j][i])
    other_information.append(other_information_i)

test_dict = {'city_type':city_type,'credit':credit,'join_month':join_month,'gat_roaming_tag':gat_roaming_tag,'provincial_roaming_tag':provincial_roaming_tag,'two_low_user_tag':two_low_user_tag,'three_low_user_tag':three_low_user_tag,'mobile_type':mobile_type,'TDLTE_tag':TDLTE_tag,'FDLTE_tag':FDLTE_tag,
             'recharge_amount':recharge_amount,'monthly_fee':monthly_fee,'grant_amount':grant_amount,'arrears_amount':arrears_amount,'over_product_voice_tag':over_product_voice_tag,'over_product_stream_income':over_product_stream_income,
             'roaming_call_duration':roaming_call_duration,'paid_call_duration':paid_call_duration,'over_product_voice_tag':over_product_voice_tag,'domestic_Long_Distance_Call_Duration':domestic_Long_Distance_Call_Duration,'gat_International_Long_Distance_Call_Duration':gat_International_Long_Distance_Call_Duration,'numbering_of_outgoing_calls':numbering_of_outgoing_calls,'numbering_of_incoming_calls':numbering_of_incoming_calls,
             'paid_Data_Traffic':paid_Data_Traffic,'free_Data_Traffic':free_Data_Traffic,'provincial_Data_Traffic':provincial_Data_Traffic,'domestic_Data_Traffic':domestic_Data_Traffic,'international_Data_Traffic':international_Data_Traffic,'data_Traffic_Used_Days':data_Traffic_Used_Days,
             'shutdown_Days':shutdown_Days,'sms_numbers':sms_numbers,'promotion_Tag':promotion_Tag,
             'churn_state_at_the_start_of_month':churn_state_at_the_start_of_month,
             'churn_state_at_the_end_of_month':churn_state_at_the_end_of_month}

test_dict_df = pd.DataFrame(test_dict)

print(test_dict_df.head())

name = 'input/'+str(n) +'_' +'data.csv'

print(name)

test_dict_df.to_csv(name)

