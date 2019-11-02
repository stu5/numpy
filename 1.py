import numpy as np
#基础运算
a = np.array([10,20,30,40])
b = np.arange(4)
print(a,b)
c = a+b
d = a*b
e = b**2
print(c)
print(d)
print(e)
f =10*np.sin(a)
print(f)
print(b)
print(b<3)
print(b==3)
##矩阵计算
import numpy as np
aa = np.array([[1,1],[0,1]])
print(aa)
bb = np.arange(4).reshape((2,2))
print(bb)
c =aa*bb #逐个相乘
print(c)
#矩阵乘法(两种方法等价)
c_dot = np.dot(aa,bb)
c_dot_2 = aa.dot(bb)
print(c_dot)
print(c_dot_2)
##随机生成2行4列0-1数字
cc = np.random.random((2,4))
print(cc)
print(np.sum(cc))
print(np.min(cc))
print(np.max(cc))
#求一行、列的最小值
print(np.min(cc,axis=1))##行
print(np.min(cc,axis=0))##列
###
A = np.arange(2,14).reshape((3,4))  ###2-14数值，3行4列
print(A)
print(np.argmin(A))####求最小值的索引
print(np.argmax(A))###最大值的索引
print(np.mean(A))####求所有值的平均值
print(np.median(A))####中位数
print(np.cumsum(A))###累加
print(np.diff(A))##累差
print(np.nonzero(A))###输出非0的位置
print(np.sort(A))##逐行排序，从小到大
print(np.transpose(A))###矩阵的反向（转置1）
print(A.T)###矩阵的反向（转置2）
print(A.T.dot(A))###矩阵乘法
print(np.clip(A,5,9))##定义小于5的值均为5，大于9的值均为9
print(np.mean(A,axis=0))##列的平均
print(np.mean(A,axis=1))##行的平均
###索引
C = np.arange(3,15)
print(C)
print(C[3])
#二维
D = np.arange(3,15).reshape((3,4))
print(D)
print(D[1][1])#输出第二行，第二列数值1
print(D[1,1])#输出第二行，第二列数值2
print(D[2,:])##输出第三行所有数
print(D[:,1])##输出第2列所有数
#迭代D的行
for row in D:
    print(row)
#迭代每一列
for column in D.T:
    print(column)
###迭代每个数
for item in D.flat:
    print(item)
print(D.flatten())
###数组合并
V = np.array([1,1,1])
print(V)
##垂直合并
N = np.array([2,2,2])
M = np.vstack((V,N))
print(V.shape,M.shape)
###水平
L = np.hstack((V,N))
print(L)
print(L.shape)
####增加维度
print(N[np.newaxis,:].shape)###行上面加了一个维度
print(N[:,np.newaxis].shape)###列上面加一个维度
print(N[:,np.newaxis])####打印出竖排的222
###多个数组合并
T = np.array([1,1,1])[:,np.newaxis]
Y = np.array([2,2,2])[:,np.newaxis]
print(T)
print(Y)
G = np.concatenate((T,Y,T),axis=0)##纵向合并
print(G)
F = np.concatenate((T,Y,T),axis=1)###横向合并
print(F)
######array数组分割
U = np.arange(12).reshape((3,4))
print(U)
print(np.split(U,2,axis=1))###分成两列
print(np.split(U,3,axis=0))###分成3行
print(np.array_split(U,3,axis=1))###不相等分割
print(np.vsplit(U,3))###纵向分割成3行
print(np.hsplit(U,2))###横向分割成2列
###数组赋值
I = np.arange(4)
print(I)
l = I
k = l
I[0]=55
print(I)
print(k)
print(k is I)
###若不想关联
k = I.copy()
I[3]=99
print(k)
print(I)
print(I is k)
