import pandas as pd
from pandas import Series,DataFrame
import xlrd

## 计算提成公式
def royalty(amount):
    result = 0
    if amount > 100000 and amount <=200000:
        result = amount * 0.05
    elif amount > 200000 and amount <=300000:
        result = amount * 0.06
    elif amount > 300000 and amount <=400000:
        result = amount * 0.07
    elif amount > 400000:
        result = amount * 0.08

    # if amount > 400000:
    #     result += ((amount - 400000) * 0.08)
    # if amount > 300000:
    #     tmp = (amount - 300000) if amount <= 400000 else 100000
    #     result += tmp * 0.07
    # if amount > 200000:
    #     tmp = (amount - 200000) if amount <= 300000 else 100000
    #     result += tmp * 0.06
    # if amount > 100000:
    #     tmp = (amount - 100000) if amount <= 200000 else 100000
    #     result += tmp * 0.05
    return result


execl_path = '/Users/lige/Downloads/12month.xlsx'
xsjbxx = '/Users/lige/Downloads/xsjbxxb.xlsx'
rymxb_path = '/Users/lige/Downloads/人员工号明细.xlsx'

### 筛选后厂理工课程开始
source_data = pd.read_excel(execl_path, sheet_name='本月业绩')
refund_data = pd.read_excel(execl_path, sheet_name='本月退费')
houchang_course = source_data[(source_data.学科=='后厂理工学院') & (source_data.流水金额>100)]
# houchang_course.to_excel('/Users/lige/caibao/output.xlsx')
### 筛选后厂理工课程结束

## 工号
wb = xlrd.open_workbook(rymxb_path)
sheet1 = wb.sheet_by_name('Sheet1')
sheet1_names = sheet1.col_values(1)
sheet1_job_numbers = sheet1.col_values(0)
job_number_dict = {}
for i in range(len(sheet1_names)):
    job_number_dict[sheet1_names[i]] = sheet1_job_numbers[i]

## 工号、入职时间、员工类型
wb = xlrd.open_workbook(execl_path)
month_sheet_11 = wb.sheet_by_name('11月提成')
col0 = month_sheet_11.col_values(1)[10:]
job_number = month_sheet_11.col_values(2)[10:]
date_entry = month_sheet_11.col_values(8)[10:]
job_type = month_sheet_11.col_values(9)[10:]
date_entry_dict = {}
job_type_dict = {}
for i in range(len(col0)):
    date_entry_dict[col0[i]] = date_entry[i]
    job_type_dict[col0[i]] = job_type[i]

print(job_number_dict)
print('--------------------')
print(date_entry_dict)
print('-------------------')
print(job_type_dict)

## 读取核算方式
wb = xlrd.open_workbook(execl_path)
hsfs_sheet = wb.sheet_by_name('核算方式')
hsfs = hsfs_sheet.col_values(0)
hsfs = hsfs[7:]
hsfs = {i.split(' ')[0]:i.split(' ')[1].replace('元', '') for i in hsfs}
# print(hsfs)

source1 = houchang_course[['销售', '流水金额']]
xiaoshou = houchang_course['销售'].values
xiaoshou = list(set(xiaoshou))
amount = []
source2 = source1.groupby('销售').sum()
xs_names = []
dep3s = []
dep4s = []
royaltys = []
positions = []
job_numbers = []
job_types = []
date_entrys = []

## 读取部门3  部门4信息
department_info = pd.read_excel(xsjbxx).values
dep_xs_names = department_info[:, 1]
dep_3 = department_info[:, 2]
dep_4 = department_info[:, 3]
position = department_info[:, 4]
dep3_dict = {}
dep4_dict = {}
position_dict = {}

## 生成退费dict
refund_data_1 = refund_data[(refund_data.流水号=='退差价') | (refund_data.流水号=='退费') | (refund_data.流水号=='转班')]
# print (refund_data.info())
refund_data_2 = refund_data_1[['贷方发生额2', '销售名称']]
rd = refund_data_2.groupby('销售名称').sum()
xsmc = refund_data_2.销售名称.values
xsmc = list(set(xsmc))
refund_xs_names = {}
refunds = []
for xs in xsmc:
    refund_xs_names[xs] = -float(rd.loc[xs]['贷方发生额2'])

for i in range(len(dep_xs_names)):
    dep3_dict[dep_xs_names[i]] = dep_3[i]
    dep4_dict[dep_xs_names[i]] = dep_4[i]
    position_dict[dep_xs_names[i]] = position[i]

for xs in xiaoshou:
    if not xs in dep_xs_names:
        continue
    xs_names.append(xs)
    a = float(source2.loc[xs]['流水金额'])
    if xs in hsfs.keys():
        print(xs)
        a += float(hsfs[xs])
    amount.append(a)
    if xs in refund_xs_names.keys():
        a -= float(refund_xs_names[xs])
        refunds.append(refund_xs_names[xs])
    else:
        refunds.append('')
    royaltys.append(royalty(a))
    if xs in dep3_dict.keys():
        dep3s.append(dep3_dict[xs])
    else:
        dep3s.append('')
    if xs in dep4_dict.keys():
        dep4s.append(dep4_dict[xs])
    else:
        dep4s.append('')
    if xs in position_dict.keys():
        positions.append(position_dict[xs])
    else:
        positions.append('课程顾问')
    # if xs in date_entry_dict.keys():
    #     date_entrys.append(date_entry_dict[xs])
    # else:
    #     date_entrys.append('')
    if xs in job_type_dict.keys():
        job_types.append(job_type_dict[xs])
    else:
        job_types.append('')
    if xs in job_number_dict.keys():
        job_numbers.append(job_number_dict[xs])
    else:
        job_numbers.append('')


tmp1 = {
    '销售' : xs_names,
    '工号' : job_numbers,
    '部门1' : ['销售中心'] * len(xs_names),
    '部门2' : ['后厂理工学院'] * len(xs_names),
    '部门3' : dep3s,
    '部门4' : dep4s,
    '岗位' : positions,
    '入职日期' : [''] * len(xs_names),
    '员工类型' : job_types,
    '当月回款金额' : amount,
    '当月退款/小白班' : [''] * len(xs_names),
    '回款总计' : amount,
    '产设' : [''] * len(xs_names),
    '提点' : [''] * len(xs_names),
    '当月提成' : royaltys,
    '其他退费' : refunds,
    '退费提点' : [''] * len(xs_names),
    '应扣提成' : [''] * len(xs_names),
    '应发合计' : [''] * len(xs_names),
    '其他应发' : [''] * len(xs_names),
    '最终发放' : [''] * len(xs_names),
    '季度结转' : [''] * len(xs_names),
    '提成类型' : [''] * len(xs_names),
    '备注' : [''] * len(xs_names),
}
output2 = DataFrame(tmp1)

print (output2)


output2.to_excel('/Users/lige/caibao/output.xlsx')