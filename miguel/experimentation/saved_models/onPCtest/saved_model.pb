Ї┘6
дє
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
*
Erf
x"T
y"T"
Ttype:
2
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
√
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%╖╤8"&
exponential_avg_factorfloat%  А?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
о
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
Ж
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( И
?
Mul
x"T
y"T
z"T"
Ttype:
2	Р

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
Н
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
┴
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring Ии
Ъ
StatelessWhile

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint

@
StaticRegexFullMatch	
input

output
"
patternstring
ў
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
░
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handleКщшelement_dtype"
element_dtypetype"

shape_typetype:
2	
Я
TensorListReserve
element_shape"
shape_type
num_elements(
handleКщшelement_dtype"
element_dtypetype"

shape_typetype:
2	
И
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint         
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.11.02v2.11.0-rc2-15-g6290819256d8ф▓2
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
О
Adam/v/node_prediction/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameAdam/v/node_prediction/bias
З
/Adam/v/node_prediction/bias/Read/ReadVariableOpReadVariableOpAdam/v/node_prediction/bias*
_output_shapes
:*
dtype0
О
Adam/m/node_prediction/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameAdam/m/node_prediction/bias
З
/Adam/m/node_prediction/bias/Read/ReadVariableOpReadVariableOpAdam/m/node_prediction/bias*
_output_shapes
:*
dtype0
Ц
Adam/v/node_prediction/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`*.
shared_nameAdam/v/node_prediction/kernel
П
1Adam/v/node_prediction/kernel/Read/ReadVariableOpReadVariableOpAdam/v/node_prediction/kernel*
_output_shapes

:`*
dtype0
Ц
Adam/m/node_prediction/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`*.
shared_nameAdam/m/node_prediction/kernel
П
1Adam/m/node_prediction/kernel/Read/ReadVariableOpReadVariableOpAdam/m/node_prediction/kernel*
_output_shapes

:`*
dtype0
Ъ
!Adam/v/layer_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*2
shared_name#!Adam/v/layer_normalization_5/beta
У
5Adam/v/layer_normalization_5/beta/Read/ReadVariableOpReadVariableOp!Adam/v/layer_normalization_5/beta*
_output_shapes
:`*
dtype0
Ъ
!Adam/m/layer_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*2
shared_name#!Adam/m/layer_normalization_5/beta
У
5Adam/m/layer_normalization_5/beta/Read/ReadVariableOpReadVariableOp!Adam/m/layer_normalization_5/beta*
_output_shapes
:`*
dtype0
Ь
"Adam/v/layer_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*3
shared_name$"Adam/v/layer_normalization_5/gamma
Х
6Adam/v/layer_normalization_5/gamma/Read/ReadVariableOpReadVariableOp"Adam/v/layer_normalization_5/gamma*
_output_shapes
:`*
dtype0
Ь
"Adam/m/layer_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*3
shared_name$"Adam/m/layer_normalization_5/gamma
Х
6Adam/m/layer_normalization_5/gamma/Read/ReadVariableOpReadVariableOp"Adam/m/layer_normalization_5/gamma*
_output_shapes
:`*
dtype0
Ж
Adam/v/mpn/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*(
shared_nameAdam/v/mpn/dense_1/bias

+Adam/v/mpn/dense_1/bias/Read/ReadVariableOpReadVariableOpAdam/v/mpn/dense_1/bias*
_output_shapes
:`*
dtype0
Ж
Adam/m/mpn/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*(
shared_nameAdam/m/mpn/dense_1/bias

+Adam/m/mpn/dense_1/bias/Read/ReadVariableOpReadVariableOpAdam/m/mpn/dense_1/bias*
_output_shapes
:`*
dtype0
П
Adam/v/mpn/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	└`**
shared_nameAdam/v/mpn/dense_1/kernel
И
-Adam/v/mpn/dense_1/kernel/Read/ReadVariableOpReadVariableOpAdam/v/mpn/dense_1/kernel*
_output_shapes
:	└`*
dtype0
П
Adam/m/mpn/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	└`**
shared_nameAdam/m/mpn/dense_1/kernel
И
-Adam/m/mpn/dense_1/kernel/Read/ReadVariableOpReadVariableOpAdam/m/mpn/dense_1/kernel*
_output_shapes
:	└`*
dtype0
Ъ
!Adam/v/layer_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*2
shared_name#!Adam/v/layer_normalization_4/beta
У
5Adam/v/layer_normalization_4/beta/Read/ReadVariableOpReadVariableOp!Adam/v/layer_normalization_4/beta*
_output_shapes
:`*
dtype0
Ъ
!Adam/m/layer_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*2
shared_name#!Adam/m/layer_normalization_4/beta
У
5Adam/m/layer_normalization_4/beta/Read/ReadVariableOpReadVariableOp!Adam/m/layer_normalization_4/beta*
_output_shapes
:`*
dtype0
Ь
"Adam/v/layer_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*3
shared_name$"Adam/v/layer_normalization_4/gamma
Х
6Adam/v/layer_normalization_4/gamma/Read/ReadVariableOpReadVariableOp"Adam/v/layer_normalization_4/gamma*
_output_shapes
:`*
dtype0
Ь
"Adam/m/layer_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*3
shared_name$"Adam/m/layer_normalization_4/gamma
Х
6Adam/m/layer_normalization_4/gamma/Read/ReadVariableOpReadVariableOp"Adam/m/layer_normalization_4/gamma*
_output_shapes
:`*
dtype0
z
Adam/v/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*"
shared_nameAdam/v/dense/bias
s
%Adam/v/dense/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense/bias*
_output_shapes
:`*
dtype0
z
Adam/m/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*"
shared_nameAdam/m/dense/bias
s
%Adam/m/dense/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense/bias*
_output_shapes
:`*
dtype0
Г
Adam/v/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	а`*$
shared_nameAdam/v/dense/kernel
|
'Adam/v/dense/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense/kernel*
_output_shapes
:	а`*
dtype0
Г
Adam/m/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	а`*$
shared_nameAdam/m/dense/kernel
|
'Adam/m/dense/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense/kernel*
_output_shapes
:	а`*
dtype0
Ъ
!Adam/v/layer_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*2
shared_name#!Adam/v/layer_normalization_3/beta
У
5Adam/v/layer_normalization_3/beta/Read/ReadVariableOpReadVariableOp!Adam/v/layer_normalization_3/beta*
_output_shapes
:`*
dtype0
Ъ
!Adam/m/layer_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*2
shared_name#!Adam/m/layer_normalization_3/beta
У
5Adam/m/layer_normalization_3/beta/Read/ReadVariableOpReadVariableOp!Adam/m/layer_normalization_3/beta*
_output_shapes
:`*
dtype0
Ь
"Adam/v/layer_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*3
shared_name$"Adam/v/layer_normalization_3/gamma
Х
6Adam/v/layer_normalization_3/gamma/Read/ReadVariableOpReadVariableOp"Adam/v/layer_normalization_3/gamma*
_output_shapes
:`*
dtype0
Ь
"Adam/m/layer_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*3
shared_name$"Adam/m/layer_normalization_3/gamma
Х
6Adam/m/layer_normalization_3/gamma/Read/ReadVariableOpReadVariableOp"Adam/m/layer_normalization_3/gamma*
_output_shapes
:`*
dtype0
Ъ
!Adam/v/layer_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*2
shared_name#!Adam/v/layer_normalization_2/beta
У
5Adam/v/layer_normalization_2/beta/Read/ReadVariableOpReadVariableOp!Adam/v/layer_normalization_2/beta*
_output_shapes
:`*
dtype0
Ъ
!Adam/m/layer_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*2
shared_name#!Adam/m/layer_normalization_2/beta
У
5Adam/m/layer_normalization_2/beta/Read/ReadVariableOpReadVariableOp!Adam/m/layer_normalization_2/beta*
_output_shapes
:`*
dtype0
Ь
"Adam/v/layer_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*3
shared_name$"Adam/v/layer_normalization_2/gamma
Х
6Adam/v/layer_normalization_2/gamma/Read/ReadVariableOpReadVariableOp"Adam/v/layer_normalization_2/gamma*
_output_shapes
:`*
dtype0
Ь
"Adam/m/layer_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*3
shared_name$"Adam/m/layer_normalization_2/gamma
Х
6Adam/m/layer_normalization_2/gamma/Read/ReadVariableOpReadVariableOp"Adam/m/layer_normalization_2/gamma*
_output_shapes
:`*
dtype0
В
Adam/v/edge_ide2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*&
shared_nameAdam/v/edge_ide2/bias
{
)Adam/v/edge_ide2/bias/Read/ReadVariableOpReadVariableOpAdam/v/edge_ide2/bias*
_output_shapes
:`*
dtype0
В
Adam/m/edge_ide2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*&
shared_nameAdam/m/edge_ide2/bias
{
)Adam/m/edge_ide2/bias/Read/ReadVariableOpReadVariableOpAdam/m/edge_ide2/bias*
_output_shapes
:`*
dtype0
К
Adam/v/edge_ide2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@`*(
shared_nameAdam/v/edge_ide2/kernel
Г
+Adam/v/edge_ide2/kernel/Read/ReadVariableOpReadVariableOpAdam/v/edge_ide2/kernel*
_output_shapes

:@`*
dtype0
К
Adam/m/edge_ide2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@`*(
shared_nameAdam/m/edge_ide2/kernel
Г
+Adam/m/edge_ide2/kernel/Read/ReadVariableOpReadVariableOpAdam/m/edge_ide2/kernel*
_output_shapes

:@`*
dtype0
Ъ
!Adam/v/layer_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/v/layer_normalization_1/beta
У
5Adam/v/layer_normalization_1/beta/Read/ReadVariableOpReadVariableOp!Adam/v/layer_normalization_1/beta*
_output_shapes
:@*
dtype0
Ъ
!Adam/m/layer_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/m/layer_normalization_1/beta
У
5Adam/m/layer_normalization_1/beta/Read/ReadVariableOpReadVariableOp!Adam/m/layer_normalization_1/beta*
_output_shapes
:@*
dtype0
Ь
"Adam/v/layer_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/v/layer_normalization_1/gamma
Х
6Adam/v/layer_normalization_1/gamma/Read/ReadVariableOpReadVariableOp"Adam/v/layer_normalization_1/gamma*
_output_shapes
:@*
dtype0
Ь
"Adam/m/layer_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/m/layer_normalization_1/gamma
Х
6Adam/m/layer_normalization_1/gamma/Read/ReadVariableOpReadVariableOp"Adam/m/layer_normalization_1/gamma*
_output_shapes
:@*
dtype0
В
Adam/v/edge_ide1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/v/edge_ide1/bias
{
)Adam/v/edge_ide1/bias/Read/ReadVariableOpReadVariableOpAdam/v/edge_ide1/bias*
_output_shapes
:@*
dtype0
В
Adam/m/edge_ide1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/m/edge_ide1/bias
{
)Adam/m/edge_ide1/bias/Read/ReadVariableOpReadVariableOpAdam/m/edge_ide1/bias*
_output_shapes
:@*
dtype0
К
Adam/v/edge_ide1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*(
shared_nameAdam/v/edge_ide1/kernel
Г
+Adam/v/edge_ide1/kernel/Read/ReadVariableOpReadVariableOpAdam/v/edge_ide1/kernel*
_output_shapes

:@*
dtype0
К
Adam/m/edge_ide1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*(
shared_nameAdam/m/edge_ide1/kernel
Г
+Adam/m/edge_ide1/kernel/Read/ReadVariableOpReadVariableOpAdam/m/edge_ide1/kernel*
_output_shapes

:@*
dtype0
В
Adam/v/node_ide2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*&
shared_nameAdam/v/node_ide2/bias
{
)Adam/v/node_ide2/bias/Read/ReadVariableOpReadVariableOpAdam/v/node_ide2/bias*
_output_shapes
:`*
dtype0
В
Adam/m/node_ide2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*&
shared_nameAdam/m/node_ide2/bias
{
)Adam/m/node_ide2/bias/Read/ReadVariableOpReadVariableOpAdam/m/node_ide2/bias*
_output_shapes
:`*
dtype0
К
Adam/v/node_ide2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@`*(
shared_nameAdam/v/node_ide2/kernel
Г
+Adam/v/node_ide2/kernel/Read/ReadVariableOpReadVariableOpAdam/v/node_ide2/kernel*
_output_shapes

:@`*
dtype0
К
Adam/m/node_ide2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@`*(
shared_nameAdam/m/node_ide2/kernel
Г
+Adam/m/node_ide2/kernel/Read/ReadVariableOpReadVariableOpAdam/m/node_ide2/kernel*
_output_shapes

:@`*
dtype0
Ц
Adam/v/layer_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!Adam/v/layer_normalization/beta
П
3Adam/v/layer_normalization/beta/Read/ReadVariableOpReadVariableOpAdam/v/layer_normalization/beta*
_output_shapes
:@*
dtype0
Ц
Adam/m/layer_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!Adam/m/layer_normalization/beta
П
3Adam/m/layer_normalization/beta/Read/ReadVariableOpReadVariableOpAdam/m/layer_normalization/beta*
_output_shapes
:@*
dtype0
Ш
 Adam/v/layer_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" Adam/v/layer_normalization/gamma
С
4Adam/v/layer_normalization/gamma/Read/ReadVariableOpReadVariableOp Adam/v/layer_normalization/gamma*
_output_shapes
:@*
dtype0
Ш
 Adam/m/layer_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" Adam/m/layer_normalization/gamma
С
4Adam/m/layer_normalization/gamma/Read/ReadVariableOpReadVariableOp Adam/m/layer_normalization/gamma*
_output_shapes
:@*
dtype0
В
Adam/v/node_ide1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/v/node_ide1/bias
{
)Adam/v/node_ide1/bias/Read/ReadVariableOpReadVariableOpAdam/v/node_ide1/bias*
_output_shapes
:@*
dtype0
В
Adam/m/node_ide1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/m/node_ide1/bias
{
)Adam/m/node_ide1/bias/Read/ReadVariableOpReadVariableOpAdam/m/node_ide1/bias*
_output_shapes
:@*
dtype0
К
Adam/v/node_ide1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*(
shared_nameAdam/v/node_ide1/kernel
Г
+Adam/v/node_ide1/kernel/Read/ReadVariableOpReadVariableOpAdam/v/node_ide1/kernel*
_output_shapes

:@*
dtype0
К
Adam/m/node_ide1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*(
shared_nameAdam/m/node_ide1/kernel
Г
+Adam/m/node_ide1/kernel/Read/ReadVariableOpReadVariableOpAdam/m/node_ide1/kernel*
_output_shapes

:@*
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
М
layer_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*+
shared_namelayer_normalization_5/beta
Е
.layer_normalization_5/beta/Read/ReadVariableOpReadVariableOplayer_normalization_5/beta*
_output_shapes
:`*
dtype0
О
layer_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*,
shared_namelayer_normalization_5/gamma
З
/layer_normalization_5/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_5/gamma*
_output_shapes
:`*
dtype0
x
mpn/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*!
shared_namempn/dense_1/bias
q
$mpn/dense_1/bias/Read/ReadVariableOpReadVariableOpmpn/dense_1/bias*
_output_shapes
:`*
dtype0
Б
mpn/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	└`*#
shared_namempn/dense_1/kernel
z
&mpn/dense_1/kernel/Read/ReadVariableOpReadVariableOpmpn/dense_1/kernel*
_output_shapes
:	└`*
dtype0
М
layer_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*+
shared_namelayer_normalization_4/beta
Е
.layer_normalization_4/beta/Read/ReadVariableOpReadVariableOplayer_normalization_4/beta*
_output_shapes
:`*
dtype0
О
layer_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*,
shared_namelayer_normalization_4/gamma
З
/layer_normalization_4/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_4/gamma*
_output_shapes
:`*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:`*
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	а`*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	а`*
dtype0
А
node_prediction/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_namenode_prediction/bias
y
(node_prediction/bias/Read/ReadVariableOpReadVariableOpnode_prediction/bias*
_output_shapes
:*
dtype0
И
node_prediction/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`*'
shared_namenode_prediction/kernel
Б
*node_prediction/kernel/Read/ReadVariableOpReadVariableOpnode_prediction/kernel*
_output_shapes

:`*
dtype0
М
layer_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*+
shared_namelayer_normalization_3/beta
Е
.layer_normalization_3/beta/Read/ReadVariableOpReadVariableOplayer_normalization_3/beta*
_output_shapes
:`*
dtype0
О
layer_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*,
shared_namelayer_normalization_3/gamma
З
/layer_normalization_3/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_3/gamma*
_output_shapes
:`*
dtype0
М
layer_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*+
shared_namelayer_normalization_2/beta
Е
.layer_normalization_2/beta/Read/ReadVariableOpReadVariableOplayer_normalization_2/beta*
_output_shapes
:`*
dtype0
О
layer_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*,
shared_namelayer_normalization_2/gamma
З
/layer_normalization_2/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_2/gamma*
_output_shapes
:`*
dtype0
t
edge_ide2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*
shared_nameedge_ide2/bias
m
"edge_ide2/bias/Read/ReadVariableOpReadVariableOpedge_ide2/bias*
_output_shapes
:`*
dtype0
|
edge_ide2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@`*!
shared_nameedge_ide2/kernel
u
$edge_ide2/kernel/Read/ReadVariableOpReadVariableOpedge_ide2/kernel*
_output_shapes

:@`*
dtype0
М
layer_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namelayer_normalization_1/beta
Е
.layer_normalization_1/beta/Read/ReadVariableOpReadVariableOplayer_normalization_1/beta*
_output_shapes
:@*
dtype0
О
layer_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namelayer_normalization_1/gamma
З
/layer_normalization_1/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_1/gamma*
_output_shapes
:@*
dtype0
t
edge_ide1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameedge_ide1/bias
m
"edge_ide1/bias/Read/ReadVariableOpReadVariableOpedge_ide1/bias*
_output_shapes
:@*
dtype0
|
edge_ide1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*!
shared_nameedge_ide1/kernel
u
$edge_ide1/kernel/Read/ReadVariableOpReadVariableOpedge_ide1/kernel*
_output_shapes

:@*
dtype0
t
node_ide2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*
shared_namenode_ide2/bias
m
"node_ide2/bias/Read/ReadVariableOpReadVariableOpnode_ide2/bias*
_output_shapes
:`*
dtype0
|
node_ide2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@`*!
shared_namenode_ide2/kernel
u
$node_ide2/kernel/Read/ReadVariableOpReadVariableOpnode_ide2/kernel*
_output_shapes

:@`*
dtype0
И
layer_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_namelayer_normalization/beta
Б
,layer_normalization/beta/Read/ReadVariableOpReadVariableOplayer_normalization/beta*
_output_shapes
:@*
dtype0
К
layer_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_namelayer_normalization/gamma
Г
-layer_normalization/gamma/Read/ReadVariableOpReadVariableOplayer_normalization/gamma*
_output_shapes
:@*
dtype0
t
node_ide1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namenode_ide1/bias
m
"node_ide1/bias/Read/ReadVariableOpReadVariableOpnode_ide1/bias*
_output_shapes
:@*
dtype0
|
node_ide1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*!
shared_namenode_ide1/kernel
u
$node_ide1/kernel/Read/ReadVariableOpReadVariableOpnode_ide1/kernel*
_output_shapes

:@*
dtype0
Ф
serving_default_input_1Placeholder*4
_output_shapes"
 :                  *
dtype0*)
shape :                  
Ф
serving_default_input_2Placeholder*4
_output_shapes"
 :                  *
dtype0*)
shape :                  
Ф
serving_default_input_3Placeholder*4
_output_shapes"
 :                  *
dtype0*)
shape :                  
Ф
serving_default_input_4Placeholder*4
_output_shapes"
 :                  *
dtype0*)
shape :                  
И
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_2serving_default_input_3serving_default_input_4node_ide1/kernelnode_ide1/biaslayer_normalization/gammalayer_normalization/betaedge_ide1/kerneledge_ide1/biasnode_ide2/kernelnode_ide2/biaslayer_normalization_1/gammalayer_normalization_1/betaedge_ide2/kerneledge_ide2/biaslayer_normalization_2/gammalayer_normalization_2/betalayer_normalization_3/gammalayer_normalization_3/betadense/kernel
dense/biaslayer_normalization_4/gammalayer_normalization_4/betampn/dense_1/kernelmpn/dense_1/biaslayer_normalization_5/gammalayer_normalization_5/betanode_prediction/kernelnode_prediction/bias*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  *<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *,
f'R%
#__inference_signature_wrapper_17411

NoOpNoOp
А░
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*║п
valueппBлп Bгп
┤
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer-9
layer_with_weights-6
layer-10
layer_with_weights-7
layer-11
layer-12
layer-13
layer-14
layer_with_weights-8
layer-15
layer_with_weights-9
layer-16
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
* 
ж
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses

!kernel
"bias*
О
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses* 
п
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses
/axis
	0gamma
1beta*
* 
ж
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses

8kernel
9bias*
ж
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses

@kernel
Abias*
п
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses
Haxis
	Igamma
Jbeta*
ж
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses

Qkernel
Rbias*

S	keras_api* 
п
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses
Zaxis
	[gamma
\beta*
п
]	variables
^trainable_variables
_regularization_losses
`	keras_api
a__call__
*b&call_and_return_all_conditional_losses
caxis
	dgamma
ebeta*
* 

f	keras_api* 
* 
┘
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses
mcombine_layer
nmessage_layer
oupdate_layer
pupdate_norm*
ж
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
u__call__
*v&call_and_return_all_conditional_losses

wkernel
xbias*
╦
!0
"1
02
13
84
95
@6
A7
I8
J9
Q10
R11
[12
\13
d14
e15
y16
z17
{18
|19
}20
~21
22
А23
w24
x25*
╦
!0
"1
02
13
84
95
@6
A7
I8
J9
Q10
R11
[12
\13
d14
e15
y16
z17
{18
|19
}20
~21
22
А23
w24
x25*
* 
╡
Бnon_trainable_variables
Вlayers
Гmetrics
 Дlayer_regularization_losses
Еlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
Жtrace_0
Зtrace_1
Иtrace_2
Йtrace_3* 
:
Кtrace_0
Лtrace_1
Мtrace_2
Нtrace_3* 
* 
И
О
_variables
П_iterations
Р_learning_rate
С_index_dict
Т
_momentums
У_velocities
Ф_update_step_xla*

Хserving_default* 

!0
"1*

!0
"1*
* 
Ш
Цnon_trainable_variables
Чlayers
Шmetrics
 Щlayer_regularization_losses
Ъlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses*

Ыtrace_0* 

Ьtrace_0* 
`Z
VARIABLE_VALUEnode_ide1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEnode_ide1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
Ц
Эnon_trainable_variables
Юlayers
Яmetrics
 аlayer_regularization_losses
бlayer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses* 
:
вtrace_0
гtrace_1
дtrace_2
еtrace_3* 
:
жtrace_0
зtrace_1
иtrace_2
йtrace_3* 

00
11*

00
11*
* 
Ш
кnon_trainable_variables
лlayers
мmetrics
 нlayer_regularization_losses
оlayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses*

пtrace_0* 

░trace_0* 
* 
hb
VARIABLE_VALUElayer_normalization/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUElayer_normalization/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*

80
91*

80
91*
* 
Ш
▒non_trainable_variables
▓layers
│metrics
 ┤layer_regularization_losses
╡layer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses*

╢trace_0* 

╖trace_0* 
`Z
VARIABLE_VALUEnode_ide2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEnode_ide2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

@0
A1*

@0
A1*
* 
Ш
╕non_trainable_variables
╣layers
║metrics
 ╗layer_regularization_losses
╝layer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses*

╜trace_0* 

╛trace_0* 
`Z
VARIABLE_VALUEedge_ide1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEedge_ide1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

I0
J1*

I0
J1*
* 
Ш
┐non_trainable_variables
└layers
┴metrics
 ┬layer_regularization_losses
├layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses*

─trace_0* 

┼trace_0* 
* 
jd
VARIABLE_VALUElayer_normalization_1/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUElayer_normalization_1/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE*

Q0
R1*

Q0
R1*
* 
Ш
╞non_trainable_variables
╟layers
╚metrics
 ╔layer_regularization_losses
╩layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses*

╦trace_0* 

╠trace_0* 
`Z
VARIABLE_VALUEedge_ide2/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEedge_ide2/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

[0
\1*

[0
\1*
* 
Ш
═non_trainable_variables
╬layers
╧metrics
 ╨layer_regularization_losses
╤layer_metrics
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses*

╥trace_0* 

╙trace_0* 
* 
jd
VARIABLE_VALUElayer_normalization_2/gamma5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUElayer_normalization_2/beta4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUE*

d0
e1*

d0
e1*
* 
Ш
╘non_trainable_variables
╒layers
╓metrics
 ╫layer_regularization_losses
╪layer_metrics
]	variables
^trainable_variables
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses*

┘trace_0* 

┌trace_0* 
* 
jd
VARIABLE_VALUElayer_normalization_3/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUElayer_normalization_3/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE*
* 
=
y0
z1
{2
|3
}4
~5
6
А7*
=
y0
z1
{2
|3
}4
~5
6
А7*
* 
Ш
█non_trainable_variables
▄layers
▌metrics
 ▐layer_regularization_losses
▀layer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses*

рtrace_0
сtrace_1* 

тtrace_0
уtrace_1* 
Ф
ф	variables
хtrainable_variables
цregularization_losses
ч	keras_api
ш__call__
+щ&call_and_return_all_conditional_losses* 
ї
ъlayer_with_weights-0
ъlayer-0
layer-1
ыlayer_with_weights-1
ыlayer-2
ь	variables
эtrainable_variables
юregularization_losses
я	keras_api
Ё__call__
+ё&call_and_return_all_conditional_losses*
м
Є	variables
єtrainable_variables
Їregularization_losses
ї	keras_api
Ў__call__
+ў&call_and_return_all_conditional_losses

}kernel
~bias*
╠
layer-0
°layer_with_weights-0
°layer-1
∙	variables
·trainable_variables
√regularization_losses
№	keras_api
¤__call__
+■&call_and_return_all_conditional_losses*

w0
x1*

w0
x1*
* 
Ш
 non_trainable_variables
Аlayers
Бmetrics
 Вlayer_regularization_losses
Гlayer_metrics
q	variables
rtrainable_variables
sregularization_losses
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses*

Дtrace_0* 

Еtrace_0* 
f`
VARIABLE_VALUEnode_prediction/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEnode_prediction/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUE
dense/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUElayer_normalization_4/gamma'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUElayer_normalization_4/beta'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEmpn/dense_1/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEmpn/dense_1/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUElayer_normalization_5/gamma'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUElayer_normalization_5/beta'variables/23/.ATTRIBUTES/VARIABLE_VALUE*
* 
В
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16*

Ж0
З1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
╫
П0
И1
Й2
К3
Л4
М5
Н6
О7
П8
Р9
С10
Т11
У12
Ф13
Х14
Ц15
Ч16
Ш17
Щ18
Ъ19
Ы20
Ь21
Э22
Ю23
Я24
а25
б26
в27
г28
д29
е30
ж31
з32
и33
й34
к35
л36
м37
н38
о39
п40
░41
▒42
▓43
│44
┤45
╡46
╢47
╖48
╕49
╣50
║51
╗52*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
ф
И0
К1
М2
О3
Р4
Т5
Ф6
Ц7
Ш8
Ъ9
Ь10
Ю11
а12
в13
д14
ж15
и16
к17
м18
о19
░20
▓21
┤22
╢23
╕24
║25*
ф
Й0
Л1
Н2
П3
С4
У5
Х6
Ч7
Щ8
Ы9
Э10
Я11
б12
г13
е14
з15
й16
л17
н18
п19
▒20
│21
╡22
╖23
╣24
╗25*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
 
m0
n1
o2
p3*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
Ь
╝non_trainable_variables
╜layers
╛metrics
 ┐layer_regularization_losses
└layer_metrics
ф	variables
хtrainable_variables
цregularization_losses
ш__call__
+щ&call_and_return_all_conditional_losses
'щ"call_and_return_conditional_losses* 
* 
* 
м
┴	variables
┬trainable_variables
├regularization_losses
─	keras_api
┼__call__
+╞&call_and_return_all_conditional_losses

ykernel
zbias*
╢
╟	variables
╚trainable_variables
╔regularization_losses
╩	keras_api
╦__call__
+╠&call_and_return_all_conditional_losses
	═axis
	{gamma
|beta*
 
y0
z1
{2
|3*
 
y0
z1
{2
|3*
* 
Ю
╬non_trainable_variables
╧layers
╨metrics
 ╤layer_regularization_losses
╥layer_metrics
ь	variables
эtrainable_variables
юregularization_losses
Ё__call__
+ё&call_and_return_all_conditional_losses
'ё"call_and_return_conditional_losses*
:
╙trace_0
╘trace_1
╒trace_2
╓trace_3* 
:
╫trace_0
╪trace_1
┘trace_2
┌trace_3* 

}0
~1*

}0
~1*
* 
Ю
█non_trainable_variables
▄layers
▌metrics
 ▐layer_regularization_losses
▀layer_metrics
Є	variables
єtrainable_variables
Їregularization_losses
Ў__call__
+ў&call_and_return_all_conditional_losses
'ў"call_and_return_conditional_losses*
* 
* 
╖
р	variables
сtrainable_variables
тregularization_losses
у	keras_api
ф__call__
+х&call_and_return_all_conditional_losses
	цaxis
	gamma
	Аbeta*

0
А1*

0
А1*
* 
Ю
чnon_trainable_variables
шlayers
щmetrics
 ъlayer_regularization_losses
ыlayer_metrics
∙	variables
·trainable_variables
√regularization_losses
¤__call__
+■&call_and_return_all_conditional_losses
'■"call_and_return_conditional_losses*
:
ьtrace_0
эtrace_1
юtrace_2
яtrace_3* 
:
Ёtrace_0
ёtrace_1
Єtrace_2
єtrace_3* 
* 
* 
* 
* 
* 
* 
* 
<
Ї	variables
ї	keras_api

Ўtotal

ўcount*
M
°	variables
∙	keras_api

·total

√count
№
_fn_kwargs*
b\
VARIABLE_VALUEAdam/m/node_ide1/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/node_ide1/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/node_ide1/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/node_ide1/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE Adam/m/layer_normalization/gamma1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE Adam/v/layer_normalization/gamma1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/m/layer_normalization/beta1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/v/layer_normalization/beta1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/node_ide2/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/node_ide2/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/node_ide2/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/node_ide2/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/m/edge_ide1/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/edge_ide1/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/edge_ide1/bias2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/edge_ide1/bias2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/layer_normalization_1/gamma2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/layer_normalization_1/gamma2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/m/layer_normalization_1/beta2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/v/layer_normalization_1/beta2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/m/edge_ide2/kernel2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/edge_ide2/kernel2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/edge_ide2/bias2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/edge_ide2/bias2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/layer_normalization_2/gamma2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/layer_normalization_2/gamma2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/m/layer_normalization_2/beta2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/v/layer_normalization_2/beta2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/layer_normalization_3/gamma2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/layer_normalization_3/gamma2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/m/layer_normalization_3/beta2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/v/layer_normalization_3/beta2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense/kernel2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense/kernel2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/m/dense/bias2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/v/dense/bias2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/layer_normalization_4/gamma2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/layer_normalization_4/gamma2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/m/layer_normalization_4/beta2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/v/layer_normalization_4/beta2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEAdam/m/mpn/dense_1/kernel2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEAdam/v/mpn/dense_1/kernel2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/m/mpn/dense_1/bias2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/mpn/dense_1/bias2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/layer_normalization_5/gamma2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/layer_normalization_5/gamma2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/m/layer_normalization_5/beta2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/v/layer_normalization_5/beta2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEAdam/m/node_prediction/kernel2optimizer/_variables/49/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEAdam/v/node_prediction/kernel2optimizer/_variables/50/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEAdam/m/node_prediction/bias2optimizer/_variables/51/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEAdam/v/node_prediction/bias2optimizer/_variables/52/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 

y0
z1*

y0
z1*
* 
Ю
¤non_trainable_variables
■layers
 metrics
 Аlayer_regularization_losses
Бlayer_metrics
┴	variables
┬trainable_variables
├regularization_losses
┼__call__
+╞&call_and_return_all_conditional_losses
'╞"call_and_return_conditional_losses*

Вtrace_0* 

Гtrace_0* 

{0
|1*

{0
|1*
* 
Ю
Дnon_trainable_variables
Еlayers
Жmetrics
 Зlayer_regularization_losses
Иlayer_metrics
╟	variables
╚trainable_variables
╔regularization_losses
╦__call__
+╠&call_and_return_all_conditional_losses
'╠"call_and_return_conditional_losses*

Йtrace_0* 

Кtrace_0* 
* 
* 

ъ0
1
ы2*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

0
А1*

0
А1*
* 
Ю
Лnon_trainable_variables
Мlayers
Нmetrics
 Оlayer_regularization_losses
Пlayer_metrics
р	variables
сtrainable_variables
тregularization_losses
ф__call__
+х&call_and_return_all_conditional_losses
'х"call_and_return_conditional_losses*

Рtrace_0* 

Сtrace_0* 
* 
* 

0
°1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

Ў0
ў1*

Ї	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

·0
√1*

°	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
№ 
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$node_ide1/kernel/Read/ReadVariableOp"node_ide1/bias/Read/ReadVariableOp-layer_normalization/gamma/Read/ReadVariableOp,layer_normalization/beta/Read/ReadVariableOp$node_ide2/kernel/Read/ReadVariableOp"node_ide2/bias/Read/ReadVariableOp$edge_ide1/kernel/Read/ReadVariableOp"edge_ide1/bias/Read/ReadVariableOp/layer_normalization_1/gamma/Read/ReadVariableOp.layer_normalization_1/beta/Read/ReadVariableOp$edge_ide2/kernel/Read/ReadVariableOp"edge_ide2/bias/Read/ReadVariableOp/layer_normalization_2/gamma/Read/ReadVariableOp.layer_normalization_2/beta/Read/ReadVariableOp/layer_normalization_3/gamma/Read/ReadVariableOp.layer_normalization_3/beta/Read/ReadVariableOp*node_prediction/kernel/Read/ReadVariableOp(node_prediction/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp/layer_normalization_4/gamma/Read/ReadVariableOp.layer_normalization_4/beta/Read/ReadVariableOp&mpn/dense_1/kernel/Read/ReadVariableOp$mpn/dense_1/bias/Read/ReadVariableOp/layer_normalization_5/gamma/Read/ReadVariableOp.layer_normalization_5/beta/Read/ReadVariableOpiteration/Read/ReadVariableOp!learning_rate/Read/ReadVariableOp+Adam/m/node_ide1/kernel/Read/ReadVariableOp+Adam/v/node_ide1/kernel/Read/ReadVariableOp)Adam/m/node_ide1/bias/Read/ReadVariableOp)Adam/v/node_ide1/bias/Read/ReadVariableOp4Adam/m/layer_normalization/gamma/Read/ReadVariableOp4Adam/v/layer_normalization/gamma/Read/ReadVariableOp3Adam/m/layer_normalization/beta/Read/ReadVariableOp3Adam/v/layer_normalization/beta/Read/ReadVariableOp+Adam/m/node_ide2/kernel/Read/ReadVariableOp+Adam/v/node_ide2/kernel/Read/ReadVariableOp)Adam/m/node_ide2/bias/Read/ReadVariableOp)Adam/v/node_ide2/bias/Read/ReadVariableOp+Adam/m/edge_ide1/kernel/Read/ReadVariableOp+Adam/v/edge_ide1/kernel/Read/ReadVariableOp)Adam/m/edge_ide1/bias/Read/ReadVariableOp)Adam/v/edge_ide1/bias/Read/ReadVariableOp6Adam/m/layer_normalization_1/gamma/Read/ReadVariableOp6Adam/v/layer_normalization_1/gamma/Read/ReadVariableOp5Adam/m/layer_normalization_1/beta/Read/ReadVariableOp5Adam/v/layer_normalization_1/beta/Read/ReadVariableOp+Adam/m/edge_ide2/kernel/Read/ReadVariableOp+Adam/v/edge_ide2/kernel/Read/ReadVariableOp)Adam/m/edge_ide2/bias/Read/ReadVariableOp)Adam/v/edge_ide2/bias/Read/ReadVariableOp6Adam/m/layer_normalization_2/gamma/Read/ReadVariableOp6Adam/v/layer_normalization_2/gamma/Read/ReadVariableOp5Adam/m/layer_normalization_2/beta/Read/ReadVariableOp5Adam/v/layer_normalization_2/beta/Read/ReadVariableOp6Adam/m/layer_normalization_3/gamma/Read/ReadVariableOp6Adam/v/layer_normalization_3/gamma/Read/ReadVariableOp5Adam/m/layer_normalization_3/beta/Read/ReadVariableOp5Adam/v/layer_normalization_3/beta/Read/ReadVariableOp'Adam/m/dense/kernel/Read/ReadVariableOp'Adam/v/dense/kernel/Read/ReadVariableOp%Adam/m/dense/bias/Read/ReadVariableOp%Adam/v/dense/bias/Read/ReadVariableOp6Adam/m/layer_normalization_4/gamma/Read/ReadVariableOp6Adam/v/layer_normalization_4/gamma/Read/ReadVariableOp5Adam/m/layer_normalization_4/beta/Read/ReadVariableOp5Adam/v/layer_normalization_4/beta/Read/ReadVariableOp-Adam/m/mpn/dense_1/kernel/Read/ReadVariableOp-Adam/v/mpn/dense_1/kernel/Read/ReadVariableOp+Adam/m/mpn/dense_1/bias/Read/ReadVariableOp+Adam/v/mpn/dense_1/bias/Read/ReadVariableOp6Adam/m/layer_normalization_5/gamma/Read/ReadVariableOp6Adam/v/layer_normalization_5/gamma/Read/ReadVariableOp5Adam/m/layer_normalization_5/beta/Read/ReadVariableOp5Adam/v/layer_normalization_5/beta/Read/ReadVariableOp1Adam/m/node_prediction/kernel/Read/ReadVariableOp1Adam/v/node_prediction/kernel/Read/ReadVariableOp/Adam/m/node_prediction/bias/Read/ReadVariableOp/Adam/v/node_prediction/bias/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*a
TinZ
X2V	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *'
f"R 
__inference__traced_save_20629
ч
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamenode_ide1/kernelnode_ide1/biaslayer_normalization/gammalayer_normalization/betanode_ide2/kernelnode_ide2/biasedge_ide1/kerneledge_ide1/biaslayer_normalization_1/gammalayer_normalization_1/betaedge_ide2/kerneledge_ide2/biaslayer_normalization_2/gammalayer_normalization_2/betalayer_normalization_3/gammalayer_normalization_3/betanode_prediction/kernelnode_prediction/biasdense/kernel
dense/biaslayer_normalization_4/gammalayer_normalization_4/betampn/dense_1/kernelmpn/dense_1/biaslayer_normalization_5/gammalayer_normalization_5/beta	iterationlearning_rateAdam/m/node_ide1/kernelAdam/v/node_ide1/kernelAdam/m/node_ide1/biasAdam/v/node_ide1/bias Adam/m/layer_normalization/gamma Adam/v/layer_normalization/gammaAdam/m/layer_normalization/betaAdam/v/layer_normalization/betaAdam/m/node_ide2/kernelAdam/v/node_ide2/kernelAdam/m/node_ide2/biasAdam/v/node_ide2/biasAdam/m/edge_ide1/kernelAdam/v/edge_ide1/kernelAdam/m/edge_ide1/biasAdam/v/edge_ide1/bias"Adam/m/layer_normalization_1/gamma"Adam/v/layer_normalization_1/gamma!Adam/m/layer_normalization_1/beta!Adam/v/layer_normalization_1/betaAdam/m/edge_ide2/kernelAdam/v/edge_ide2/kernelAdam/m/edge_ide2/biasAdam/v/edge_ide2/bias"Adam/m/layer_normalization_2/gamma"Adam/v/layer_normalization_2/gamma!Adam/m/layer_normalization_2/beta!Adam/v/layer_normalization_2/beta"Adam/m/layer_normalization_3/gamma"Adam/v/layer_normalization_3/gamma!Adam/m/layer_normalization_3/beta!Adam/v/layer_normalization_3/betaAdam/m/dense/kernelAdam/v/dense/kernelAdam/m/dense/biasAdam/v/dense/bias"Adam/m/layer_normalization_4/gamma"Adam/v/layer_normalization_4/gamma!Adam/m/layer_normalization_4/beta!Adam/v/layer_normalization_4/betaAdam/m/mpn/dense_1/kernelAdam/v/mpn/dense_1/kernelAdam/m/mpn/dense_1/biasAdam/v/mpn/dense_1/bias"Adam/m/layer_normalization_5/gamma"Adam/v/layer_normalization_5/gamma!Adam/m/layer_normalization_5/beta!Adam/v/layer_normalization_5/betaAdam/m/node_prediction/kernelAdam/v/node_prediction/kernelAdam/m/node_prediction/biasAdam/v/node_prediction/biastotal_1count_1totalcount*`
TinY
W2U*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В **
f%R#
!__inference__traced_restore_20891ТО/
┌
╧
*__inference_sequential_layer_call_fn_15431
dense_input
unknown:	а`
	unknown_0:`
	unknown_1:`
	unknown_2:`
identityИвStatefulPartitionedCallЖ
StatefulPartitionedCallStatefulPartitionedCalldense_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  `*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_15420|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  ``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):                  а: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:b ^
5
_output_shapes#
!:                  а
%
_user_specified_namedense_input
Ў
Ц
)__inference_edge_ide1_layer_call_fn_18978

inputs
unknown:@
	unknown_0:@
identityИвStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_edge_ide1_layer_call_and_return_conditional_losses_15839|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:                  : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
Г
√
D__inference_node_ide1_layer_call_and_return_conditional_losses_15737

inputs3
!tensordot_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвTensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ┐
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ь
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:В
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*4
_output_shapes"
 :                  К
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:                  К
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : з
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:М
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Е
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  @l
IdentityIdentityBiasAdd:output:0^NoOp*
T0*4
_output_shapes"
 :                  @z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:                  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
МX
ё
E__inference_sequential_layer_call_and_return_conditional_losses_19991

inputs:
'dense_tensordot_readvariableop_resource:	а`3
%dense_biasadd_readvariableop_resource:`A
3layer_normalization_4_mul_3_readvariableop_resource:`?
1layer_normalization_4_add_readvariableop_resource:`
identityИвdense/BiasAdd/ReadVariableOpвdense/Tensordot/ReadVariableOpв(layer_normalization_4/add/ReadVariableOpв*layer_normalization_4/mul_3/ReadVariableOpЗ
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes
:	а`*
dtype0^
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:e
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       K
dense/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:_
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╙
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ╫
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:_
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: А
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: a
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Ж
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ]
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ┤
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Л
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:П
dense/Tensordot/transpose	Transposeinputsdense/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:                  аЬ
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  Ь
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `a
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:`_
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ┐
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Ю
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  `~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0Ч
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `V
lambda/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Й
lambda/Gelu/mulMullambda/Gelu/mul/x:output:0dense/BiasAdd:output:0*
T0*4
_output_shapes"
 :                  `W
lambda/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *є╡?Т
lambda/Gelu/truedivRealDivdense/BiasAdd:output:0lambda/Gelu/Cast/x:output:0*
T0*4
_output_shapes"
 :                  `n
lambda/Gelu/ErfErflambda/Gelu/truediv:z:0*
T0*4
_output_shapes"
 :                  `V
lambda/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?И
lambda/Gelu/addAddV2lambda/Gelu/add/x:output:0lambda/Gelu/Erf:y:0*
T0*4
_output_shapes"
 :                  `Б
lambda/Gelu/mul_1Mullambda/Gelu/mul:z:0lambda/Gelu/add:z:0*
T0*4
_output_shapes"
 :                  ``
layer_normalization_4/ShapeShapelambda/Gelu/mul_1:z:0*
T0*
_output_shapes
:s
)layer_normalization_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+layer_normalization_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┐
#layer_normalization_4/strided_sliceStridedSlice$layer_normalization_4/Shape:output:02layer_normalization_4/strided_slice/stack:output:04layer_normalization_4/strided_slice/stack_1:output:04layer_normalization_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization_4/mul/xConst*
_output_shapes
: *
dtype0*
value	B :Х
layer_normalization_4/mulMul$layer_normalization_4/mul/x:output:0,layer_normalization_4/strided_slice:output:0*
T0*
_output_shapes
: u
+layer_normalization_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╟
%layer_normalization_4/strided_slice_1StridedSlice$layer_normalization_4/Shape:output:04layer_normalization_4/strided_slice_1/stack:output:06layer_normalization_4/strided_slice_1/stack_1:output:06layer_normalization_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskТ
layer_normalization_4/mul_1Mullayer_normalization_4/mul:z:0.layer_normalization_4/strided_slice_1:output:0*
T0*
_output_shapes
: u
+layer_normalization_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╟
%layer_normalization_4/strided_slice_2StridedSlice$layer_normalization_4/Shape:output:04layer_normalization_4/strided_slice_2/stack:output:06layer_normalization_4/strided_slice_2/stack_1:output:06layer_normalization_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
layer_normalization_4/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :Ы
layer_normalization_4/mul_2Mul&layer_normalization_4/mul_2/x:output:0.layer_normalization_4/strided_slice_2:output:0*
T0*
_output_shapes
: g
%layer_normalization_4/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :g
%layer_normalization_4/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :√
#layer_normalization_4/Reshape/shapePack.layer_normalization_4/Reshape/shape/0:output:0layer_normalization_4/mul_1:z:0layer_normalization_4/mul_2:z:0.layer_normalization_4/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:з
layer_normalization_4/ReshapeReshapelambda/Gelu/mul_1:z:0,layer_normalization_4/Reshape/shape:output:0*
T0*/
_output_shapes
:         `x
!layer_normalization_4/ones/packedPacklayer_normalization_4/mul_1:z:0*
N*
T0*
_output_shapes
:e
 layer_normalization_4/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?з
layer_normalization_4/onesFill*layer_normalization_4/ones/packed:output:0)layer_normalization_4/ones/Const:output:0*
T0*#
_output_shapes
:         y
"layer_normalization_4/zeros/packedPacklayer_normalization_4/mul_1:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    к
layer_normalization_4/zerosFill+layer_normalization_4/zeros/packed:output:0*layer_normalization_4/zeros/Const:output:0*
T0*#
_output_shapes
:         ^
layer_normalization_4/ConstConst*
_output_shapes
: *
dtype0*
valueB `
layer_normalization_4/Const_1Const*
_output_shapes
: *
dtype0*
valueB з
&layer_normalization_4/FusedBatchNormV3FusedBatchNormV3&layer_normalization_4/Reshape:output:0#layer_normalization_4/ones:output:0$layer_normalization_4/zeros:output:0$layer_normalization_4/Const:output:0&layer_normalization_4/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:         `:         :         :         :         :*
data_formatNCHW*
epsilon%oГ:╗
layer_normalization_4/Reshape_1Reshape*layer_normalization_4/FusedBatchNormV3:y:0$layer_normalization_4/Shape:output:0*
T0*4
_output_shapes"
 :                  `Ъ
*layer_normalization_4/mul_3/ReadVariableOpReadVariableOp3layer_normalization_4_mul_3_readvariableop_resource*
_output_shapes
:`*
dtype0┐
layer_normalization_4/mul_3Mul(layer_normalization_4/Reshape_1:output:02layer_normalization_4/mul_3/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `Ц
(layer_normalization_4/add/ReadVariableOpReadVariableOp1layer_normalization_4_add_readvariableop_resource*
_output_shapes
:`*
dtype0┤
layer_normalization_4/addAddV2layer_normalization_4/mul_3:z:00layer_normalization_4/add/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `y
IdentityIdentitylayer_normalization_4/add:z:0^NoOp*
T0*4
_output_shapes"
 :                  `▐
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp)^layer_normalization_4/add/ReadVariableOp+^layer_normalization_4/mul_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):                  а: : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2T
(layer_normalization_4/add/ReadVariableOp(layer_normalization_4/add/ReadVariableOp2X
*layer_normalization_4/mul_3/ReadVariableOp*layer_normalization_4/mul_3/ReadVariableOp:] Y
5
_output_shapes#
!:                  а
 
_user_specified_nameinputs
К
Ю
5__inference_layer_normalization_2_layer_call_fn_19112

inputs
unknown:`
	unknown_0:`
identityИвStatefulPartitionedCallЄ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  `*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_layer_normalization_2_layer_call_and_return_conditional_losses_16024|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  ``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:                  `: : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :                  `
 
_user_specified_nameinputs
°?
Х
model_mpn_scan_while_body_15137:
6model_mpn_scan_while_model_mpn_scan_while_loop_counter5
1model_mpn_scan_while_model_mpn_scan_strided_slice$
 model_mpn_scan_while_placeholder&
"model_mpn_scan_while_placeholder_1&
"model_mpn_scan_while_placeholder_29
5model_mpn_scan_while_model_mpn_scan_strided_slice_1_0u
qmodel_mpn_scan_while_tensorarrayv2read_tensorlistgetitem_model_mpn_scan_tensorarrayunstack_tensorlistfromtensor_0y
umodel_mpn_scan_while_tensorarrayv2read_1_tensorlistgetitem_model_mpn_scan_tensorarrayunstack_1_tensorlistfromtensor_0y
umodel_mpn_scan_while_tensorarrayv2read_2_tensorlistgetitem_model_mpn_scan_tensorarrayunstack_2_tensorlistfromtensor_0E
Amodel_mpn_scan_while_unsortedsegmentsum_model_mpn_strided_slice_0!
model_mpn_scan_while_identity#
model_mpn_scan_while_identity_1#
model_mpn_scan_while_identity_2#
model_mpn_scan_while_identity_3#
model_mpn_scan_while_identity_47
3model_mpn_scan_while_model_mpn_scan_strided_slice_1s
omodel_mpn_scan_while_tensorarrayv2read_tensorlistgetitem_model_mpn_scan_tensorarrayunstack_tensorlistfromtensorw
smodel_mpn_scan_while_tensorarrayv2read_1_tensorlistgetitem_model_mpn_scan_tensorarrayunstack_1_tensorlistfromtensorw
smodel_mpn_scan_while_tensorarrayv2read_2_tensorlistgetitem_model_mpn_scan_tensorarrayunstack_2_tensorlistfromtensorC
?model_mpn_scan_while_unsortedsegmentsum_model_mpn_strided_sliceЧ
Fmodel/mpn/scan/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    `   ё
8model/mpn/scan/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemqmodel_mpn_scan_while_tensorarrayv2read_tensorlistgetitem_model_mpn_scan_tensorarrayunstack_tensorlistfromtensor_0 model_mpn_scan_while_placeholderOmodel/mpn/scan/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         `*
element_dtype0Щ
Hmodel/mpn/scan/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ∙
:model/mpn/scan/while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItemumodel_mpn_scan_while_tensorarrayv2read_1_tensorlistgetitem_model_mpn_scan_tensorarrayunstack_1_tensorlistfromtensor_0 model_mpn_scan_while_placeholderQmodel/mpn/scan/while/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Щ
Hmodel/mpn/scan/while/TensorArrayV2Read_2/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ∙
:model/mpn/scan/while/TensorArrayV2Read_2/TensorListGetItemTensorListGetItemumodel_mpn_scan_while_tensorarrayv2read_2_tensorlistgetitem_model_mpn_scan_tensorarrayunstack_2_tensorlistfromtensor_0 model_mpn_scan_while_placeholderQmodel/mpn/scan/while/TensorArrayV2Read_2/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0y
(model/mpn/scan/while/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*model/mpn/scan/while/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       {
*model/mpn/scan/while/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      є
"model/mpn/scan/while/strided_sliceStridedSliceAmodel/mpn/scan/while/TensorArrayV2Read_2/TensorListGetItem:item:01model/mpn/scan/while/strided_slice/stack:output:03model/mpn/scan/while/strided_slice/stack_1:output:03model/mpn/scan/while/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask┐
model/mpn/scan/while/mulMul?model/mpn/scan/while/TensorArrayV2Read/TensorListGetItem:item:0+model/mpn/scan/while/strided_slice:output:0*
T0*'
_output_shapes
:         `{
*model/mpn/scan/while/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       }
,model/mpn/scan/while/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       }
,model/mpn/scan/while/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      П
$model/mpn/scan/while/strided_slice_1StridedSliceAmodel/mpn/scan/while/TensorArrayV2Read_1/TensorListGetItem:item:03model/mpn/scan/while/strided_slice_1/stack:output:05model/mpn/scan/while/strided_slice_1/stack_1:output:05model/mpn/scan/while/strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_maskП
'model/mpn/scan/while/UnsortedSegmentSumUnsortedSegmentSummodel/mpn/scan/while/mul:z:0-model/mpn/scan/while/strided_slice_1:output:0Amodel_mpn_scan_while_unsortedsegmentsum_model_mpn_strided_slice_0*
T0*
Tindices0*'
_output_shapes
:         `Ж
9model/mpn/scan/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem"model_mpn_scan_while_placeholder_2 model_mpn_scan_while_placeholder0model/mpn/scan/while/UnsortedSegmentSum:output:0*
_output_shapes
: *
element_dtype0:щш╥\
model/mpn/scan/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Й
model/mpn/scan/while/addAddV2 model_mpn_scan_while_placeholder#model/mpn/scan/while/add/y:output:0*
T0*
_output_shapes
: ^
model/mpn/scan/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :г
model/mpn/scan/while/add_1AddV26model_mpn_scan_while_model_mpn_scan_while_loop_counter%model/mpn/scan/while/add_1/y:output:0*
T0*
_output_shapes
: j
model/mpn/scan/while/IdentityIdentitymodel/mpn/scan/while/add_1:z:0*
T0*
_output_shapes
: 
model/mpn/scan/while/Identity_1Identity1model_mpn_scan_while_model_mpn_scan_strided_slice*
T0*
_output_shapes
: j
model/mpn/scan/while/Identity_2Identitymodel/mpn/scan/while/add:z:0*
T0*
_output_shapes
: П
model/mpn/scan/while/Identity_3Identity0model/mpn/scan/while/UnsortedSegmentSum:output:0*
T0*'
_output_shapes
:         `Ч
model/mpn/scan/while/Identity_4IdentityImodel/mpn/scan/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: "G
model_mpn_scan_while_identity&model/mpn/scan/while/Identity:output:0"K
model_mpn_scan_while_identity_1(model/mpn/scan/while/Identity_1:output:0"K
model_mpn_scan_while_identity_2(model/mpn/scan/while/Identity_2:output:0"K
model_mpn_scan_while_identity_3(model/mpn/scan/while/Identity_3:output:0"K
model_mpn_scan_while_identity_4(model/mpn/scan/while/Identity_4:output:0"l
3model_mpn_scan_while_model_mpn_scan_strided_slice_15model_mpn_scan_while_model_mpn_scan_strided_slice_1_0"ь
smodel_mpn_scan_while_tensorarrayv2read_1_tensorlistgetitem_model_mpn_scan_tensorarrayunstack_1_tensorlistfromtensorumodel_mpn_scan_while_tensorarrayv2read_1_tensorlistgetitem_model_mpn_scan_tensorarrayunstack_1_tensorlistfromtensor_0"ь
smodel_mpn_scan_while_tensorarrayv2read_2_tensorlistgetitem_model_mpn_scan_tensorarrayunstack_2_tensorlistfromtensorumodel_mpn_scan_while_tensorarrayv2read_2_tensorlistgetitem_model_mpn_scan_tensorarrayunstack_2_tensorlistfromtensor_0"ф
omodel_mpn_scan_while_tensorarrayv2read_tensorlistgetitem_model_mpn_scan_tensorarrayunstack_tensorlistfromtensorqmodel_mpn_scan_while_tensorarrayv2read_tensorlistgetitem_model_mpn_scan_tensorarrayunstack_tensorlistfromtensor_0"Д
?model_mpn_scan_while_unsortedsegmentsum_model_mpn_strided_sliceAmodel_mpn_scan_while_unsortedsegmentsum_model_mpn_strided_slice_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : :         `: : : : : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         `:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
Ў
Ц
)__inference_edge_ide2_layer_call_fn_19073

inputs
unknown:@`
	unknown_0:`
identityИвStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  `*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_edge_ide2_layer_call_and_return_conditional_losses_15966|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  ``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:                  @: : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :                  @
 
_user_specified_nameinputs
╝	
]
A__inference_lambda_layer_call_and_return_conditional_losses_18838

inputs
identityO

Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?k
Gelu/mulMulGelu/mul/x:output:0inputs*
T0*4
_output_shapes"
 :                  `P
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *є╡?t
Gelu/truedivRealDivinputsGelu/Cast/x:output:0*
T0*4
_output_shapes"
 :                  ``
Gelu/ErfErfGelu/truediv:z:0*
T0*4
_output_shapes"
 :                  `O

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?s
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*4
_output_shapes"
 :                  `l

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*4
_output_shapes"
 :                  `c
IdentityIdentityGelu/mul_1:z:0*
T0*4
_output_shapes"
 :                  `"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :                  `:\ X
4
_output_shapes"
 :                  `
 
_user_specified_nameinputs
Р%
є
P__inference_layer_normalization_4_layer_call_and_return_conditional_losses_15413

inputs+
mul_3_readvariableop_resource:`)
add_readvariableop_resource:`
identityИвadd/ReadVariableOpвmul_3/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/xConst*
_output_shapes
: *
dtype0*
value	B :S
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: _
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
mul_1Mulmul:z:0strided_slice_1:output:0*
T0*
_output_shapes
: _
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_2Mulmul_2/x:output:0strided_slice_2:output:0*
T0*
_output_shapes
: Q
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Н
Reshape/shapePackReshape/shape/0:output:0	mul_1:z:0	mul_2:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:         `L
ones/packedPack	mul_1:z:0*
N*
T0*
_output_shapes
:O

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?e
onesFillones/packed:output:0ones/Const:output:0*
T0*#
_output_shapes
:         M
zeros/packedPack	mul_1:z:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    h
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*#
_output_shapes
:         H
ConstConst*
_output_shapes
: *
dtype0*
valueB J
Const_1Const*
_output_shapes
: *
dtype0*
valueB г
FusedBatchNormV3FusedBatchNormV3Reshape:output:0ones:output:0zeros:output:0Const:output:0Const_1:output:0*
T0*
U0*o
_output_shapes]
[:         `:         :         :         :         :*
data_formatNCHW*
epsilon%oГ:y
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*4
_output_shapes"
 :                  `n
mul_3/ReadVariableOpReadVariableOpmul_3_readvariableop_resource*
_output_shapes
:`*
dtype0}
mul_3MulReshape_1:output:0mul_3/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:`*
dtype0r
addAddV2	mul_3:z:0add/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `c
IdentityIdentityadd:z:0^NoOp*
T0*4
_output_shapes"
 :                  `r
NoOpNoOp^add/ReadVariableOp^mul_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:                  `: : 2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_3/ReadVariableOpmul_3/ReadVariableOp:\ X
4
_output_shapes"
 :                  `
 
_user_specified_nameinputs
Р%
є
P__inference_layer_normalization_4_layer_call_and_return_conditional_losses_20295

inputs+
mul_3_readvariableop_resource:`)
add_readvariableop_resource:`
identityИвadd/ReadVariableOpвmul_3/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/xConst*
_output_shapes
: *
dtype0*
value	B :S
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: _
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
mul_1Mulmul:z:0strided_slice_1:output:0*
T0*
_output_shapes
: _
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_2Mulmul_2/x:output:0strided_slice_2:output:0*
T0*
_output_shapes
: Q
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Н
Reshape/shapePackReshape/shape/0:output:0	mul_1:z:0	mul_2:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:         `L
ones/packedPack	mul_1:z:0*
N*
T0*
_output_shapes
:O

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?e
onesFillones/packed:output:0ones/Const:output:0*
T0*#
_output_shapes
:         M
zeros/packedPack	mul_1:z:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    h
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*#
_output_shapes
:         H
ConstConst*
_output_shapes
: *
dtype0*
valueB J
Const_1Const*
_output_shapes
: *
dtype0*
valueB г
FusedBatchNormV3FusedBatchNormV3Reshape:output:0ones:output:0zeros:output:0Const:output:0Const_1:output:0*
T0*
U0*o
_output_shapes]
[:         `:         :         :         :         :*
data_formatNCHW*
epsilon%oГ:y
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*4
_output_shapes"
 :                  `n
mul_3/ReadVariableOpReadVariableOpmul_3_readvariableop_resource*
_output_shapes
:`*
dtype0}
mul_3MulReshape_1:output:0mul_3/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:`*
dtype0r
addAddV2	mul_3:z:0add/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `c
IdentityIdentityadd:z:0^NoOp*
T0*4
_output_shapes"
 :                  `r
NoOpNoOp^add/ReadVariableOp^mul_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:                  `: : 2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_3/ReadVariableOpmul_3/ReadVariableOp:\ X
4
_output_shapes"
 :                  `
 
_user_specified_nameinputs
о
╗
%__inference_model_layer_call_fn_17179
input_1
input_2
input_3
input_4
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@`
	unknown_6:`
	unknown_7:@
	unknown_8:@
	unknown_9:@`

unknown_10:`

unknown_11:`

unknown_12:`

unknown_13:`

unknown_14:`

unknown_15:	а`

unknown_16:`

unknown_17:`

unknown_18:`

unknown_19:	└`

unknown_20:`

unknown_21:`

unknown_22:`

unknown_23:`

unknown_24:
identityИвStatefulPartitionedCall╚
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2input_3input_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  *<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_17064|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*╔
_input_shapes╖
┤:                  :                  :                  :                  : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
4
_output_shapes"
 :                  
!
_user_specified_name	input_1:]Y
4
_output_shapes"
 :                  
!
_user_specified_name	input_2:]Y
4
_output_shapes"
 :                  
!
_user_specified_name	input_3:]Y
4
_output_shapes"
 :                  
!
_user_specified_name	input_4
Й
Б
J__inference_node_prediction_layer_call_and_return_conditional_losses_19884

inputs3
!tensordot_readvariableop_resource:`-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвTensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:`*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ┐
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ь
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:В
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*4
_output_shapes"
 :                  `К
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:                  К
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         [
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : з
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:М
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Е
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  l
IdentityIdentityBiasAdd:output:0^NoOp*
T0*4
_output_shapes"
 :                  z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:                  `: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:\ X
4
_output_shapes"
 :                  `
 
_user_specified_nameinputs
О%
ё
N__inference_layer_normalization_layer_call_and_return_conditional_losses_18930

inputs+
mul_3_readvariableop_resource:@)
add_readvariableop_resource:@
identityИвadd/ReadVariableOpвmul_3/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/xConst*
_output_shapes
: *
dtype0*
value	B :S
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: _
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
mul_1Mulmul:z:0strided_slice_1:output:0*
T0*
_output_shapes
: _
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_2Mulmul_2/x:output:0strided_slice_2:output:0*
T0*
_output_shapes
: Q
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Н
Reshape/shapePackReshape/shape/0:output:0	mul_1:z:0	mul_2:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:         @L
ones/packedPack	mul_1:z:0*
N*
T0*
_output_shapes
:O

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?e
onesFillones/packed:output:0ones/Const:output:0*
T0*#
_output_shapes
:         M
zeros/packedPack	mul_1:z:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    h
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*#
_output_shapes
:         H
ConstConst*
_output_shapes
: *
dtype0*
valueB J
Const_1Const*
_output_shapes
: *
dtype0*
valueB г
FusedBatchNormV3FusedBatchNormV3Reshape:output:0ones:output:0zeros:output:0Const:output:0Const_1:output:0*
T0*
U0*o
_output_shapes]
[:         @:         :         :         :         :*
data_formatNCHW*
epsilon%oГ:y
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*4
_output_shapes"
 :                  @n
mul_3/ReadVariableOpReadVariableOpmul_3_readvariableop_resource*
_output_shapes
:@*
dtype0}
mul_3MulReshape_1:output:0mul_3/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  @j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:@*
dtype0r
addAddV2	mul_3:z:0add/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  @c
IdentityIdentityadd:z:0^NoOp*
T0*4
_output_shapes"
 :                  @r
NoOpNoOp^add/ReadVariableOp^mul_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:                  @: : 2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_3/ReadVariableOpmul_3/ReadVariableOp:\ X
4
_output_shapes"
 :                  @
 
_user_specified_nameinputs
Г
√
D__inference_node_ide2_layer_call_and_return_conditional_losses_15875

inputs3
!tensordot_readvariableop_resource:@`-
biasadd_readvariableop_resource:`
identityИвBiasAdd/ReadVariableOpвTensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@`*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ┐
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ь
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:В
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*4
_output_shapes"
 :                  @К
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:                  К
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:`Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : з
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:М
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  `r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:`*
dtype0Е
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `l
IdentityIdentityBiasAdd:output:0^NoOp*
T0*4
_output_shapes"
 :                  `z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:                  @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:\ X
4
_output_shapes"
 :                  @
 
_user_specified_nameinputs
Ў
Ц
)__inference_node_ide2_layer_call_fn_18939

inputs
unknown:@`
	unknown_0:`
identityИвStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  `*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_node_ide2_layer_call_and_return_conditional_losses_15875|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  ``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:                  @: : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :                  @
 
_user_specified_nameinputs
┌
╧
*__inference_sequential_layer_call_fn_15528
dense_input
unknown:	а`
	unknown_0:`
	unknown_1:`
	unknown_2:`
identityИвStatefulPartitionedCallЖ
StatefulPartitionedCallStatefulPartitionedCalldense_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  `*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_15504|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  ``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):                  а: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:b ^
5
_output_shapes#
!:                  а
%
_user_specified_namedense_input
вМ
Е	
>__inference_mpn_layer_call_and_return_conditional_losses_19845
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4E
2sequential_dense_tensordot_readvariableop_resource:	а`>
0sequential_dense_biasadd_readvariableop_resource:`L
>sequential_layer_normalization_4_mul_3_readvariableop_resource:`J
<sequential_layer_normalization_4_add_readvariableop_resource:`<
)dense_1_tensordot_readvariableop_resource:	└`5
'dense_1_biasadd_readvariableop_resource:`N
@sequential_1_layer_normalization_5_mul_3_readvariableop_resource:`L
>sequential_1_layer_normalization_5_add_readvariableop_resource:`
identity

identity_1

identity_2

identity_3

identity_4Ивdense_1/BiasAdd/ReadVariableOpв dense_1/Tensordot/ReadVariableOpв'sequential/dense/BiasAdd/ReadVariableOpв)sequential/dense/Tensordot/ReadVariableOpв3sequential/layer_normalization_4/add/ReadVariableOpв5sequential/layer_normalization_4/mul_3/ReadVariableOpв5sequential_1/layer_normalization_5/add/ReadVariableOpв7sequential_1/layer_normalization_5/mul_3/ReadVariableOp=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Shape_1Shapeinputs_2*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Shape_2Shapeinputs_0*
T0*
_output_shapes
:_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_2StridedSliceShape_2:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskO
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :╕
GatherV2GatherV2inputs_0inputs_2GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*8
_output_shapes&
$:"                  `*

batch_dimsR
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :└С
Reshape/shapePackstrided_slice_2:output:0strided_slice_1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:}
ReshapeReshapeGatherV2:output:0Reshape/shape:output:0*
T0*5
_output_shapes#
!:                  └V
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
         Н
concatConcatV2Reshape:output:0inputs_1concat/axis:output:0*
N*
T0*5
_output_shapes#
!:                  аЭ
)sequential/dense/Tensordot/ReadVariableOpReadVariableOp2sequential_dense_tensordot_readvariableop_resource*
_output_shapes
:	а`*
dtype0i
sequential/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:p
sequential/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       _
 sequential/dense/Tensordot/ShapeShapeconcat:output:0*
T0*
_output_shapes
:j
(sequential/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :  
#sequential/dense/Tensordot/GatherV2GatherV2)sequential/dense/Tensordot/Shape:output:0(sequential/dense/Tensordot/free:output:01sequential/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
*sequential/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Г
%sequential/dense/Tensordot/GatherV2_1GatherV2)sequential/dense/Tensordot/Shape:output:0(sequential/dense/Tensordot/axes:output:03sequential/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:j
 sequential/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: б
sequential/dense/Tensordot/ProdProd,sequential/dense/Tensordot/GatherV2:output:0)sequential/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: l
"sequential/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: з
!sequential/dense/Tensordot/Prod_1Prod.sequential/dense/Tensordot/GatherV2_1:output:0+sequential/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: h
&sequential/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : р
!sequential/dense/Tensordot/concatConcatV2(sequential/dense/Tensordot/free:output:0(sequential/dense/Tensordot/axes:output:0/sequential/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:м
 sequential/dense/Tensordot/stackPack(sequential/dense/Tensordot/Prod:output:0*sequential/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:о
$sequential/dense/Tensordot/transpose	Transposeconcat:output:0*sequential/dense/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:                  а╜
"sequential/dense/Tensordot/ReshapeReshape(sequential/dense/Tensordot/transpose:y:0)sequential/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  ╜
!sequential/dense/Tensordot/MatMulMatMul+sequential/dense/Tensordot/Reshape:output:01sequential/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `l
"sequential/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:`j
(sequential/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ы
#sequential/dense/Tensordot/concat_1ConcatV2,sequential/dense/Tensordot/GatherV2:output:0+sequential/dense/Tensordot/Const_2:output:01sequential/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:┐
sequential/dense/TensordotReshape+sequential/dense/Tensordot/MatMul:product:0,sequential/dense/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  `Ф
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0╕
sequential/dense/BiasAddBiasAdd#sequential/dense/Tensordot:output:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `a
sequential/lambda/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?к
sequential/lambda/Gelu/mulMul%sequential/lambda/Gelu/mul/x:output:0!sequential/dense/BiasAdd:output:0*
T0*4
_output_shapes"
 :                  `b
sequential/lambda/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *є╡?│
sequential/lambda/Gelu/truedivRealDiv!sequential/dense/BiasAdd:output:0&sequential/lambda/Gelu/Cast/x:output:0*
T0*4
_output_shapes"
 :                  `Д
sequential/lambda/Gelu/ErfErf"sequential/lambda/Gelu/truediv:z:0*
T0*4
_output_shapes"
 :                  `a
sequential/lambda/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?й
sequential/lambda/Gelu/addAddV2%sequential/lambda/Gelu/add/x:output:0sequential/lambda/Gelu/Erf:y:0*
T0*4
_output_shapes"
 :                  `в
sequential/lambda/Gelu/mul_1Mulsequential/lambda/Gelu/mul:z:0sequential/lambda/Gelu/add:z:0*
T0*4
_output_shapes"
 :                  `v
&sequential/layer_normalization_4/ShapeShape sequential/lambda/Gelu/mul_1:z:0*
T0*
_output_shapes
:~
4sequential/layer_normalization_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: А
6sequential/layer_normalization_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:А
6sequential/layer_normalization_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ў
.sequential/layer_normalization_4/strided_sliceStridedSlice/sequential/layer_normalization_4/Shape:output:0=sequential/layer_normalization_4/strided_slice/stack:output:0?sequential/layer_normalization_4/strided_slice/stack_1:output:0?sequential/layer_normalization_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
&sequential/layer_normalization_4/mul/xConst*
_output_shapes
: *
dtype0*
value	B :╢
$sequential/layer_normalization_4/mulMul/sequential/layer_normalization_4/mul/x:output:07sequential/layer_normalization_4/strided_slice:output:0*
T0*
_output_shapes
: А
6sequential/layer_normalization_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:В
8sequential/layer_normalization_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:В
8sequential/layer_normalization_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:■
0sequential/layer_normalization_4/strided_slice_1StridedSlice/sequential/layer_normalization_4/Shape:output:0?sequential/layer_normalization_4/strided_slice_1/stack:output:0Asequential/layer_normalization_4/strided_slice_1/stack_1:output:0Asequential/layer_normalization_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask│
&sequential/layer_normalization_4/mul_1Mul(sequential/layer_normalization_4/mul:z:09sequential/layer_normalization_4/strided_slice_1:output:0*
T0*
_output_shapes
: А
6sequential/layer_normalization_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:В
8sequential/layer_normalization_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:В
8sequential/layer_normalization_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:■
0sequential/layer_normalization_4/strided_slice_2StridedSlice/sequential/layer_normalization_4/Shape:output:0?sequential/layer_normalization_4/strided_slice_2/stack:output:0Asequential/layer_normalization_4/strided_slice_2/stack_1:output:0Asequential/layer_normalization_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(sequential/layer_normalization_4/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :╝
&sequential/layer_normalization_4/mul_2Mul1sequential/layer_normalization_4/mul_2/x:output:09sequential/layer_normalization_4/strided_slice_2:output:0*
T0*
_output_shapes
: r
0sequential/layer_normalization_4/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :r
0sequential/layer_normalization_4/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :▓
.sequential/layer_normalization_4/Reshape/shapePack9sequential/layer_normalization_4/Reshape/shape/0:output:0*sequential/layer_normalization_4/mul_1:z:0*sequential/layer_normalization_4/mul_2:z:09sequential/layer_normalization_4/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:╚
(sequential/layer_normalization_4/ReshapeReshape sequential/lambda/Gelu/mul_1:z:07sequential/layer_normalization_4/Reshape/shape:output:0*
T0*/
_output_shapes
:         `О
,sequential/layer_normalization_4/ones/packedPack*sequential/layer_normalization_4/mul_1:z:0*
N*
T0*
_output_shapes
:p
+sequential/layer_normalization_4/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╚
%sequential/layer_normalization_4/onesFill5sequential/layer_normalization_4/ones/packed:output:04sequential/layer_normalization_4/ones/Const:output:0*
T0*#
_output_shapes
:         П
-sequential/layer_normalization_4/zeros/packedPack*sequential/layer_normalization_4/mul_1:z:0*
N*
T0*
_output_shapes
:q
,sequential/layer_normalization_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ╦
&sequential/layer_normalization_4/zerosFill6sequential/layer_normalization_4/zeros/packed:output:05sequential/layer_normalization_4/zeros/Const:output:0*
T0*#
_output_shapes
:         i
&sequential/layer_normalization_4/ConstConst*
_output_shapes
: *
dtype0*
valueB k
(sequential/layer_normalization_4/Const_1Const*
_output_shapes
: *
dtype0*
valueB щ
1sequential/layer_normalization_4/FusedBatchNormV3FusedBatchNormV31sequential/layer_normalization_4/Reshape:output:0.sequential/layer_normalization_4/ones:output:0/sequential/layer_normalization_4/zeros:output:0/sequential/layer_normalization_4/Const:output:01sequential/layer_normalization_4/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:         `:         :         :         :         :*
data_formatNCHW*
epsilon%oГ:▄
*sequential/layer_normalization_4/Reshape_1Reshape5sequential/layer_normalization_4/FusedBatchNormV3:y:0/sequential/layer_normalization_4/Shape:output:0*
T0*4
_output_shapes"
 :                  `░
5sequential/layer_normalization_4/mul_3/ReadVariableOpReadVariableOp>sequential_layer_normalization_4_mul_3_readvariableop_resource*
_output_shapes
:`*
dtype0р
&sequential/layer_normalization_4/mul_3Mul3sequential/layer_normalization_4/Reshape_1:output:0=sequential/layer_normalization_4/mul_3/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `м
3sequential/layer_normalization_4/add/ReadVariableOpReadVariableOp<sequential_layer_normalization_4_add_readvariableop_resource*
_output_shapes
:`*
dtype0╒
$sequential/layer_normalization_4/addAddV2*sequential/layer_normalization_4/mul_3:z:0;sequential/layer_normalization_4/add/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `}
mulMul(sequential/layer_normalization_4/add:z:0inputs_3*
T0*4
_output_shapes"
 :                  `P
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :`s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         `A

scan/ShapeShapemul:z:0*
T0*
_output_shapes
:b
scan/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: d
scan/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:d
scan/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ъ
scan/strided_sliceStridedSlicescan/Shape:output:0!scan/strided_slice/stack:output:0#scan/strided_slice/stack_1:output:0#scan/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskq
 scan/TensorArrayV2/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    `   ┴
scan/TensorArrayV2TensorListReserve)scan/TensorArrayV2/element_shape:output:0scan/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥s
"scan/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ┼
scan/TensorArrayV2_1TensorListReserve+scan/TensorArrayV2_1/element_shape:output:0scan/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╧s
"scan/TensorArrayV2_2/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ┼
scan/TensorArrayV2_2TensorListReserve+scan/TensorArrayV2_2/element_shape:output:0scan/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Л
:scan/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    `   ф
,scan/TensorArrayUnstack/TensorListFromTensorTensorListFromTensormul:z:0Cscan/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Н
<scan/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       щ
.scan/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensorinputs_2Escan/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╧Н
<scan/TensorArrayUnstack_2/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       щ
.scan/TensorArrayUnstack_2/TensorListFromTensorTensorListFromTensorinputs_4Escan/TensorArrayUnstack_2/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥s
"scan/TensorArrayV2_3/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    `   ┼
scan/TensorArrayV2_3TensorListReserve+scan/TensorArrayV2_3/element_shape:output:0scan/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥L

scan/ConstConst*
_output_shapes
: *
dtype0*
value	B : Y
scan/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ▀

scan/whileStatelessWhile scan/while/loop_counter:output:0scan/strided_slice:output:0scan/Const:output:0zeros:output:0scan/TensorArrayV2_3:handle:0scan/strided_slice:output:0<scan/TensorArrayUnstack/TensorListFromTensor:output_handle:0>scan/TensorArrayUnstack_1/TensorListFromTensor:output_handle:0>scan/TensorArrayUnstack_2/TensorListFromTensor:output_handle:0strided_slice:output:0*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : :         `: : : : : : * 
_read_only_resource_inputs
 *
_stateful_parallelism( *!
bodyR
scan_while_body_19695*!
condR
scan_while_cond_19694*8
output_shapes'
%: : : :         `: : : : : : Ж
5scan/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    `   ┌
'scan/TensorArrayV2Stack/TensorListStackTensorListStackscan/while:output:4>scan/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  `*
element_dtype0_
lambda_1/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
         ┐
lambda_1/concatConcatV2inputs_00scan/TensorArrayV2Stack/TensorListStack:tensor:0lambda_1/concat/axis:output:0*
N*
T0*5
_output_shapes#
!:                  └Л
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource*
_output_shapes
:	└`*
dtype0`
dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       _
dense_1/Tensordot/ShapeShapelambda_1/concat:output:0*
T0*
_output_shapes
:a
dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : █
dense_1/Tensordot/GatherV2GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/free:output:0(dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ▀
dense_1/Tensordot/GatherV2_1GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/axes:output:0*dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: Ж
dense_1/Tensordot/ProdProd#dense_1/Tensordot/GatherV2:output:0 dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: М
dense_1/Tensordot/Prod_1Prod%dense_1/Tensordot/GatherV2_1:output:0"dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ╝
dense_1/Tensordot/concatConcatV2dense_1/Tensordot/free:output:0dense_1/Tensordot/axes:output:0&dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:С
dense_1/Tensordot/stackPackdense_1/Tensordot/Prod:output:0!dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:е
dense_1/Tensordot/transpose	Transposelambda_1/concat:output:0!dense_1/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:                  └в
dense_1/Tensordot/ReshapeReshapedense_1/Tensordot/transpose:y:0 dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  в
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0(dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `c
dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:`a
dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ╟
dense_1/Tensordot/concat_1ConcatV2#dense_1/Tensordot/GatherV2:output:0"dense_1/Tensordot/Const_2:output:0(dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:д
dense_1/TensordotReshape"dense_1/Tensordot/MatMul:product:0#dense_1/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  `В
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0Э
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `c
sequential_1/lambda/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?е
sequential_1/lambda/Gelu/mulMul'sequential_1/lambda/Gelu/mul/x:output:0dense_1/BiasAdd:output:0*
T0*4
_output_shapes"
 :                  `d
sequential_1/lambda/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *є╡?о
 sequential_1/lambda/Gelu/truedivRealDivdense_1/BiasAdd:output:0(sequential_1/lambda/Gelu/Cast/x:output:0*
T0*4
_output_shapes"
 :                  `И
sequential_1/lambda/Gelu/ErfErf$sequential_1/lambda/Gelu/truediv:z:0*
T0*4
_output_shapes"
 :                  `c
sequential_1/lambda/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?п
sequential_1/lambda/Gelu/addAddV2'sequential_1/lambda/Gelu/add/x:output:0 sequential_1/lambda/Gelu/Erf:y:0*
T0*4
_output_shapes"
 :                  `и
sequential_1/lambda/Gelu/mul_1Mul sequential_1/lambda/Gelu/mul:z:0 sequential_1/lambda/Gelu/add:z:0*
T0*4
_output_shapes"
 :                  `z
(sequential_1/layer_normalization_5/ShapeShape"sequential_1/lambda/Gelu/mul_1:z:0*
T0*
_output_shapes
:А
6sequential_1/layer_normalization_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: В
8sequential_1/layer_normalization_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:В
8sequential_1/layer_normalization_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:А
0sequential_1/layer_normalization_5/strided_sliceStridedSlice1sequential_1/layer_normalization_5/Shape:output:0?sequential_1/layer_normalization_5/strided_slice/stack:output:0Asequential_1/layer_normalization_5/strided_slice/stack_1:output:0Asequential_1/layer_normalization_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(sequential_1/layer_normalization_5/mul/xConst*
_output_shapes
: *
dtype0*
value	B :╝
&sequential_1/layer_normalization_5/mulMul1sequential_1/layer_normalization_5/mul/x:output:09sequential_1/layer_normalization_5/strided_slice:output:0*
T0*
_output_shapes
: В
8sequential_1/layer_normalization_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:Д
:sequential_1/layer_normalization_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Д
:sequential_1/layer_normalization_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
2sequential_1/layer_normalization_5/strided_slice_1StridedSlice1sequential_1/layer_normalization_5/Shape:output:0Asequential_1/layer_normalization_5/strided_slice_1/stack:output:0Csequential_1/layer_normalization_5/strided_slice_1/stack_1:output:0Csequential_1/layer_normalization_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask╣
(sequential_1/layer_normalization_5/mul_1Mul*sequential_1/layer_normalization_5/mul:z:0;sequential_1/layer_normalization_5/strided_slice_1:output:0*
T0*
_output_shapes
: В
8sequential_1/layer_normalization_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:Д
:sequential_1/layer_normalization_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Д
:sequential_1/layer_normalization_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
2sequential_1/layer_normalization_5/strided_slice_2StridedSlice1sequential_1/layer_normalization_5/Shape:output:0Asequential_1/layer_normalization_5/strided_slice_2/stack:output:0Csequential_1/layer_normalization_5/strided_slice_2/stack_1:output:0Csequential_1/layer_normalization_5/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
*sequential_1/layer_normalization_5/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :┬
(sequential_1/layer_normalization_5/mul_2Mul3sequential_1/layer_normalization_5/mul_2/x:output:0;sequential_1/layer_normalization_5/strided_slice_2:output:0*
T0*
_output_shapes
: t
2sequential_1/layer_normalization_5/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :t
2sequential_1/layer_normalization_5/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :╝
0sequential_1/layer_normalization_5/Reshape/shapePack;sequential_1/layer_normalization_5/Reshape/shape/0:output:0,sequential_1/layer_normalization_5/mul_1:z:0,sequential_1/layer_normalization_5/mul_2:z:0;sequential_1/layer_normalization_5/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:╬
*sequential_1/layer_normalization_5/ReshapeReshape"sequential_1/lambda/Gelu/mul_1:z:09sequential_1/layer_normalization_5/Reshape/shape:output:0*
T0*/
_output_shapes
:         `Т
.sequential_1/layer_normalization_5/ones/packedPack,sequential_1/layer_normalization_5/mul_1:z:0*
N*
T0*
_output_shapes
:r
-sequential_1/layer_normalization_5/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╬
'sequential_1/layer_normalization_5/onesFill7sequential_1/layer_normalization_5/ones/packed:output:06sequential_1/layer_normalization_5/ones/Const:output:0*
T0*#
_output_shapes
:         У
/sequential_1/layer_normalization_5/zeros/packedPack,sequential_1/layer_normalization_5/mul_1:z:0*
N*
T0*
_output_shapes
:s
.sequential_1/layer_normalization_5/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ╤
(sequential_1/layer_normalization_5/zerosFill8sequential_1/layer_normalization_5/zeros/packed:output:07sequential_1/layer_normalization_5/zeros/Const:output:0*
T0*#
_output_shapes
:         k
(sequential_1/layer_normalization_5/ConstConst*
_output_shapes
: *
dtype0*
valueB m
*sequential_1/layer_normalization_5/Const_1Const*
_output_shapes
: *
dtype0*
valueB ї
3sequential_1/layer_normalization_5/FusedBatchNormV3FusedBatchNormV33sequential_1/layer_normalization_5/Reshape:output:00sequential_1/layer_normalization_5/ones:output:01sequential_1/layer_normalization_5/zeros:output:01sequential_1/layer_normalization_5/Const:output:03sequential_1/layer_normalization_5/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:         `:         :         :         :         :*
data_formatNCHW*
epsilon%oГ:т
,sequential_1/layer_normalization_5/Reshape_1Reshape7sequential_1/layer_normalization_5/FusedBatchNormV3:y:01sequential_1/layer_normalization_5/Shape:output:0*
T0*4
_output_shapes"
 :                  `┤
7sequential_1/layer_normalization_5/mul_3/ReadVariableOpReadVariableOp@sequential_1_layer_normalization_5_mul_3_readvariableop_resource*
_output_shapes
:`*
dtype0ц
(sequential_1/layer_normalization_5/mul_3Mul5sequential_1/layer_normalization_5/Reshape_1:output:0?sequential_1/layer_normalization_5/mul_3/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `░
5sequential_1/layer_normalization_5/add/ReadVariableOpReadVariableOp>sequential_1_layer_normalization_5_add_readvariableop_resource*
_output_shapes
:`*
dtype0█
&sequential_1/layer_normalization_5/addAddV2,sequential_1/layer_normalization_5/mul_3:z:0=sequential_1/layer_normalization_5/add/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `Ж
IdentityIdentity*sequential_1/layer_normalization_5/add:z:0^NoOp*
T0*4
_output_shapes"
 :                  `e

Identity_1Identitymul:z:0^NoOp*
T0*4
_output_shapes"
 :                  `f

Identity_2Identityinputs_2^NoOp*
T0*4
_output_shapes"
 :                  f

Identity_3Identityinputs_3^NoOp*
T0*4
_output_shapes"
 :                  f

Identity_4Identityinputs_4^NoOp*
T0*4
_output_shapes"
 :                  └
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp*^sequential/dense/Tensordot/ReadVariableOp4^sequential/layer_normalization_4/add/ReadVariableOp6^sequential/layer_normalization_4/mul_3/ReadVariableOp6^sequential_1/layer_normalization_5/add/ReadVariableOp8^sequential_1/layer_normalization_5/mul_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*┼
_input_shapes│
░:                  `:                  `:                  :                  :                  : : : : : : : : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_1/Tensordot/ReadVariableOp dense_1/Tensordot/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2V
)sequential/dense/Tensordot/ReadVariableOp)sequential/dense/Tensordot/ReadVariableOp2j
3sequential/layer_normalization_4/add/ReadVariableOp3sequential/layer_normalization_4/add/ReadVariableOp2n
5sequential/layer_normalization_4/mul_3/ReadVariableOp5sequential/layer_normalization_4/mul_3/ReadVariableOp2n
5sequential_1/layer_normalization_5/add/ReadVariableOp5sequential_1/layer_normalization_5/add/ReadVariableOp2r
7sequential_1/layer_normalization_5/mul_3/ReadVariableOp7sequential_1/layer_normalization_5/mul_3/ReadVariableOp:^ Z
4
_output_shapes"
 :                  `
"
_user_specified_name
inputs_0:^Z
4
_output_shapes"
 :                  `
"
_user_specified_name
inputs_1:^Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_2:^Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_3:^Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_4
ъ5
╦
scan_while_body_16211&
"scan_while_scan_while_loop_counter!
scan_while_scan_strided_slice
scan_while_placeholder
scan_while_placeholder_1
scan_while_placeholder_2%
!scan_while_scan_strided_slice_1_0a
]scan_while_tensorarrayv2read_tensorlistgetitem_scan_tensorarrayunstack_tensorlistfromtensor_0e
ascan_while_tensorarrayv2read_1_tensorlistgetitem_scan_tensorarrayunstack_1_tensorlistfromtensor_0e
ascan_while_tensorarrayv2read_2_tensorlistgetitem_scan_tensorarrayunstack_2_tensorlistfromtensor_01
-scan_while_unsortedsegmentsum_strided_slice_0
scan_while_identity
scan_while_identity_1
scan_while_identity_2
scan_while_identity_3
scan_while_identity_4#
scan_while_scan_strided_slice_1_
[scan_while_tensorarrayv2read_tensorlistgetitem_scan_tensorarrayunstack_tensorlistfromtensorc
_scan_while_tensorarrayv2read_1_tensorlistgetitem_scan_tensorarrayunstack_1_tensorlistfromtensorc
_scan_while_tensorarrayv2read_2_tensorlistgetitem_scan_tensorarrayunstack_2_tensorlistfromtensor/
+scan_while_unsortedsegmentsum_strided_sliceН
<scan/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    `   ┐
.scan/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem]scan_while_tensorarrayv2read_tensorlistgetitem_scan_tensorarrayunstack_tensorlistfromtensor_0scan_while_placeholderEscan/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         `*
element_dtype0П
>scan/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╟
0scan/while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItemascan_while_tensorarrayv2read_1_tensorlistgetitem_scan_tensorarrayunstack_1_tensorlistfromtensor_0scan_while_placeholderGscan/while/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0П
>scan/while/TensorArrayV2Read_2/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╟
0scan/while/TensorArrayV2Read_2/TensorListGetItemTensorListGetItemascan_while_tensorarrayv2read_2_tensorlistgetitem_scan_tensorarrayunstack_2_tensorlistfromtensor_0scan_while_placeholderGscan/while/TensorArrayV2Read_2/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0o
scan/while/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       q
 scan/while/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       q
 scan/while/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ┴
scan/while/strided_sliceStridedSlice7scan/while/TensorArrayV2Read_2/TensorListGetItem:item:0'scan/while/strided_slice/stack:output:0)scan/while/strided_slice/stack_1:output:0)scan/while/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_maskб
scan/while/mulMul5scan/while/TensorArrayV2Read/TensorListGetItem:item:0!scan/while/strided_slice:output:0*
T0*'
_output_shapes
:         `q
 scan/while/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       s
"scan/while/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       s
"scan/while/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ▌
scan/while/strided_slice_1StridedSlice7scan/while/TensorArrayV2Read_1/TensorListGetItem:item:0)scan/while/strided_slice_1/stack:output:0+scan/while/strided_slice_1/stack_1:output:0+scan/while/strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask▌
scan/while/UnsortedSegmentSumUnsortedSegmentSumscan/while/mul:z:0#scan/while/strided_slice_1:output:0-scan_while_unsortedsegmentsum_strided_slice_0*
T0*
Tindices0*'
_output_shapes
:         `▐
/scan/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemscan_while_placeholder_2scan_while_placeholder&scan/while/UnsortedSegmentSum:output:0*
_output_shapes
: *
element_dtype0:щш╥R
scan/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :k
scan/while/addAddV2scan_while_placeholderscan/while/add/y:output:0*
T0*
_output_shapes
: T
scan/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :{
scan/while/add_1AddV2"scan_while_scan_while_loop_counterscan/while/add_1/y:output:0*
T0*
_output_shapes
: V
scan/while/IdentityIdentityscan/while/add_1:z:0*
T0*
_output_shapes
: a
scan/while/Identity_1Identityscan_while_scan_strided_slice*
T0*
_output_shapes
: V
scan/while/Identity_2Identityscan/while/add:z:0*
T0*
_output_shapes
: {
scan/while/Identity_3Identity&scan/while/UnsortedSegmentSum:output:0*
T0*'
_output_shapes
:         `Г
scan/while/Identity_4Identity?scan/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: "3
scan_while_identityscan/while/Identity:output:0"7
scan_while_identity_1scan/while/Identity_1:output:0"7
scan_while_identity_2scan/while/Identity_2:output:0"7
scan_while_identity_3scan/while/Identity_3:output:0"7
scan_while_identity_4scan/while/Identity_4:output:0"D
scan_while_scan_strided_slice_1!scan_while_scan_strided_slice_1_0"─
_scan_while_tensorarrayv2read_1_tensorlistgetitem_scan_tensorarrayunstack_1_tensorlistfromtensorascan_while_tensorarrayv2read_1_tensorlistgetitem_scan_tensorarrayunstack_1_tensorlistfromtensor_0"─
_scan_while_tensorarrayv2read_2_tensorlistgetitem_scan_tensorarrayunstack_2_tensorlistfromtensorascan_while_tensorarrayv2read_2_tensorlistgetitem_scan_tensorarrayunstack_2_tensorlistfromtensor_0"╝
[scan_while_tensorarrayv2read_tensorlistgetitem_scan_tensorarrayunstack_tensorlistfromtensor]scan_while_tensorarrayv2read_tensorlistgetitem_scan_tensorarrayunstack_tensorlistfromtensor_0"\
+scan_while_unsortedsegmentsum_strided_slice-scan_while_unsortedsegmentsum_strided_slice_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : :         `: : : : : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         `:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
╔
ь
G__inference_sequential_1_layer_call_and_return_conditional_losses_15620

inputs)
layer_normalization_5_15614:`)
layer_normalization_5_15616:`
identityИв-layer_normalization_5/StatefulPartitionedCall└
lambda/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  `* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_lambda_layer_call_and_return_conditional_losses_15364╟
-layer_normalization_5/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0layer_normalization_5_15614layer_normalization_5_15616*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  `*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_layer_normalization_5_layer_call_and_return_conditional_losses_15613Т
IdentityIdentity6layer_normalization_5/StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  `v
NoOpNoOp.^layer_normalization_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:                  `: : 2^
-layer_normalization_5/StatefulPartitionedCall-layer_normalization_5/StatefulPartitionedCall:\ X
4
_output_shapes"
 :                  `
 
_user_specified_nameinputs
й
╩
E__inference_sequential_layer_call_and_return_conditional_losses_15558
dense_input
dense_15546:	а`
dense_15548:`)
layer_normalization_4_15552:`)
layer_normalization_4_15554:`
identityИвdense/StatefulPartitionedCallв-layer_normalization_4/StatefulPartitionedCallє
dense/StatefulPartitionedCallStatefulPartitionedCalldense_inputdense_15546dense_15548*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  `*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_15346р
lambda/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  `* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_lambda_layer_call_and_return_conditional_losses_15461╟
-layer_normalization_4/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0layer_normalization_4_15552layer_normalization_4_15554*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  `*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_layer_normalization_4_layer_call_and_return_conditional_losses_15413Т
IdentityIdentity6layer_normalization_4/StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  `Ц
NoOpNoOp^dense/StatefulPartitionedCall.^layer_normalization_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):                  а: : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2^
-layer_normalization_4/StatefulPartitionedCall-layer_normalization_4/StatefulPartitionedCall:b ^
5
_output_shapes#
!:                  а
%
_user_specified_namedense_input
╝	
]
A__inference_lambda_layer_call_and_return_conditional_losses_15461

inputs
identityO

Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?k
Gelu/mulMulGelu/mul/x:output:0inputs*
T0*4
_output_shapes"
 :                  `P
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *є╡?t
Gelu/truedivRealDivinputsGelu/Cast/x:output:0*
T0*4
_output_shapes"
 :                  ``
Gelu/ErfErfGelu/truediv:z:0*
T0*4
_output_shapes"
 :                  `O

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?s
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*4
_output_shapes"
 :                  `l

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*4
_output_shapes"
 :                  `c
IdentityIdentityGelu/mul_1:z:0*
T0*4
_output_shapes"
 :                  `"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :                  `:\ X
4
_output_shapes"
 :                  `
 
_user_specified_nameinputs
║
┐
%__inference_model_layer_call_fn_17531
inputs_0
inputs_1
inputs_2
inputs_3
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@`
	unknown_6:`
	unknown_7:@
	unknown_8:@
	unknown_9:@`

unknown_10:`

unknown_11:`

unknown_12:`

unknown_13:`

unknown_14:`

unknown_15:	а`

unknown_16:`

unknown_17:`

unknown_18:`

unknown_19:	└`

unknown_20:`

unknown_21:`

unknown_22:`

unknown_23:`

unknown_24:
identityИвStatefulPartitionedCall╠
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2inputs_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  *<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_17064|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*╔
_input_shapes╖
┤:                  :                  :                  :                  : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_0:^Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_1:^Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_2:^Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_3
Ё9
╧	
mpn_scan_while_body_17977.
*mpn_scan_while_mpn_scan_while_loop_counter)
%mpn_scan_while_mpn_scan_strided_slice
mpn_scan_while_placeholder 
mpn_scan_while_placeholder_1 
mpn_scan_while_placeholder_2-
)mpn_scan_while_mpn_scan_strided_slice_1_0i
empn_scan_while_tensorarrayv2read_tensorlistgetitem_mpn_scan_tensorarrayunstack_tensorlistfromtensor_0m
impn_scan_while_tensorarrayv2read_1_tensorlistgetitem_mpn_scan_tensorarrayunstack_1_tensorlistfromtensor_0m
impn_scan_while_tensorarrayv2read_2_tensorlistgetitem_mpn_scan_tensorarrayunstack_2_tensorlistfromtensor_09
5mpn_scan_while_unsortedsegmentsum_mpn_strided_slice_0
mpn_scan_while_identity
mpn_scan_while_identity_1
mpn_scan_while_identity_2
mpn_scan_while_identity_3
mpn_scan_while_identity_4+
'mpn_scan_while_mpn_scan_strided_slice_1g
cmpn_scan_while_tensorarrayv2read_tensorlistgetitem_mpn_scan_tensorarrayunstack_tensorlistfromtensork
gmpn_scan_while_tensorarrayv2read_1_tensorlistgetitem_mpn_scan_tensorarrayunstack_1_tensorlistfromtensork
gmpn_scan_while_tensorarrayv2read_2_tensorlistgetitem_mpn_scan_tensorarrayunstack_2_tensorlistfromtensor7
3mpn_scan_while_unsortedsegmentsum_mpn_strided_sliceС
@mpn/scan/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    `   ╙
2mpn/scan/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemempn_scan_while_tensorarrayv2read_tensorlistgetitem_mpn_scan_tensorarrayunstack_tensorlistfromtensor_0mpn_scan_while_placeholderImpn/scan/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         `*
element_dtype0У
Bmpn/scan/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       █
4mpn/scan/while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItemimpn_scan_while_tensorarrayv2read_1_tensorlistgetitem_mpn_scan_tensorarrayunstack_1_tensorlistfromtensor_0mpn_scan_while_placeholderKmpn/scan/while/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0У
Bmpn/scan/while/TensorArrayV2Read_2/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       █
4mpn/scan/while/TensorArrayV2Read_2/TensorListGetItemTensorListGetItemimpn_scan_while_tensorarrayv2read_2_tensorlistgetitem_mpn_scan_tensorarrayunstack_2_tensorlistfromtensor_0mpn_scan_while_placeholderKmpn/scan/while/TensorArrayV2Read_2/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0s
"mpn/scan/while/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       u
$mpn/scan/while/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       u
$mpn/scan/while/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╒
mpn/scan/while/strided_sliceStridedSlice;mpn/scan/while/TensorArrayV2Read_2/TensorListGetItem:item:0+mpn/scan/while/strided_slice/stack:output:0-mpn/scan/while/strided_slice/stack_1:output:0-mpn/scan/while/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_maskн
mpn/scan/while/mulMul9mpn/scan/while/TensorArrayV2Read/TensorListGetItem:item:0%mpn/scan/while/strided_slice:output:0*
T0*'
_output_shapes
:         `u
$mpn/scan/while/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       w
&mpn/scan/while/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       w
&mpn/scan/while/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ё
mpn/scan/while/strided_slice_1StridedSlice;mpn/scan/while/TensorArrayV2Read_1/TensorListGetItem:item:0-mpn/scan/while/strided_slice_1/stack:output:0/mpn/scan/while/strided_slice_1/stack_1:output:0/mpn/scan/while/strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_maskё
!mpn/scan/while/UnsortedSegmentSumUnsortedSegmentSummpn/scan/while/mul:z:0'mpn/scan/while/strided_slice_1:output:05mpn_scan_while_unsortedsegmentsum_mpn_strided_slice_0*
T0*
Tindices0*'
_output_shapes
:         `ю
3mpn/scan/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemmpn_scan_while_placeholder_2mpn_scan_while_placeholder*mpn/scan/while/UnsortedSegmentSum:output:0*
_output_shapes
: *
element_dtype0:щш╥V
mpn/scan/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :w
mpn/scan/while/addAddV2mpn_scan_while_placeholdermpn/scan/while/add/y:output:0*
T0*
_output_shapes
: X
mpn/scan/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Л
mpn/scan/while/add_1AddV2*mpn_scan_while_mpn_scan_while_loop_countermpn/scan/while/add_1/y:output:0*
T0*
_output_shapes
: ^
mpn/scan/while/IdentityIdentitympn/scan/while/add_1:z:0*
T0*
_output_shapes
: m
mpn/scan/while/Identity_1Identity%mpn_scan_while_mpn_scan_strided_slice*
T0*
_output_shapes
: ^
mpn/scan/while/Identity_2Identitympn/scan/while/add:z:0*
T0*
_output_shapes
: Г
mpn/scan/while/Identity_3Identity*mpn/scan/while/UnsortedSegmentSum:output:0*
T0*'
_output_shapes
:         `Л
mpn/scan/while/Identity_4IdentityCmpn/scan/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: ";
mpn_scan_while_identity mpn/scan/while/Identity:output:0"?
mpn_scan_while_identity_1"mpn/scan/while/Identity_1:output:0"?
mpn_scan_while_identity_2"mpn/scan/while/Identity_2:output:0"?
mpn_scan_while_identity_3"mpn/scan/while/Identity_3:output:0"?
mpn_scan_while_identity_4"mpn/scan/while/Identity_4:output:0"T
'mpn_scan_while_mpn_scan_strided_slice_1)mpn_scan_while_mpn_scan_strided_slice_1_0"╘
gmpn_scan_while_tensorarrayv2read_1_tensorlistgetitem_mpn_scan_tensorarrayunstack_1_tensorlistfromtensorimpn_scan_while_tensorarrayv2read_1_tensorlistgetitem_mpn_scan_tensorarrayunstack_1_tensorlistfromtensor_0"╘
gmpn_scan_while_tensorarrayv2read_2_tensorlistgetitem_mpn_scan_tensorarrayunstack_2_tensorlistfromtensorimpn_scan_while_tensorarrayv2read_2_tensorlistgetitem_mpn_scan_tensorarrayunstack_2_tensorlistfromtensor_0"╠
cmpn_scan_while_tensorarrayv2read_tensorlistgetitem_mpn_scan_tensorarrayunstack_tensorlistfromtensorempn_scan_while_tensorarrayv2read_tensorlistgetitem_mpn_scan_tensorarrayunstack_tensorlistfromtensor_0"l
3mpn_scan_while_unsortedsegmentsum_mpn_strided_slice5mpn_scan_while_unsortedsegmentsum_mpn_strided_slice_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : :         `: : : : : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         `:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
Р%
є
P__inference_layer_normalization_3_layer_call_and_return_conditional_losses_16077

inputs+
mul_3_readvariableop_resource:`)
add_readvariableop_resource:`
identityИвadd/ReadVariableOpвmul_3/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/xConst*
_output_shapes
: *
dtype0*
value	B :S
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: _
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
mul_1Mulmul:z:0strided_slice_1:output:0*
T0*
_output_shapes
: _
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_2Mulmul_2/x:output:0strided_slice_2:output:0*
T0*
_output_shapes
: Q
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Н
Reshape/shapePackReshape/shape/0:output:0	mul_1:z:0	mul_2:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:         `L
ones/packedPack	mul_1:z:0*
N*
T0*
_output_shapes
:O

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?e
onesFillones/packed:output:0ones/Const:output:0*
T0*#
_output_shapes
:         M
zeros/packedPack	mul_1:z:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    h
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*#
_output_shapes
:         H
ConstConst*
_output_shapes
: *
dtype0*
valueB J
Const_1Const*
_output_shapes
: *
dtype0*
valueB г
FusedBatchNormV3FusedBatchNormV3Reshape:output:0ones:output:0zeros:output:0Const:output:0Const_1:output:0*
T0*
U0*o
_output_shapes]
[:         `:         :         :         :         :*
data_formatNCHW*
epsilon%oГ:y
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*4
_output_shapes"
 :                  `n
mul_3/ReadVariableOpReadVariableOpmul_3_readvariableop_resource*
_output_shapes
:`*
dtype0}
mul_3MulReshape_1:output:0mul_3/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:`*
dtype0r
addAddV2	mul_3:z:0add/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `c
IdentityIdentityadd:z:0^NoOp*
T0*4
_output_shapes"
 :                  `r
NoOpNoOp^add/ReadVariableOp^mul_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:                  `: : 2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_3/ReadVariableOpmul_3/ReadVariableOp:\ X
4
_output_shapes"
 :                  `
 
_user_specified_nameinputs
ъ5
╦
scan_while_body_16643&
"scan_while_scan_while_loop_counter!
scan_while_scan_strided_slice
scan_while_placeholder
scan_while_placeholder_1
scan_while_placeholder_2%
!scan_while_scan_strided_slice_1_0a
]scan_while_tensorarrayv2read_tensorlistgetitem_scan_tensorarrayunstack_tensorlistfromtensor_0e
ascan_while_tensorarrayv2read_1_tensorlistgetitem_scan_tensorarrayunstack_1_tensorlistfromtensor_0e
ascan_while_tensorarrayv2read_2_tensorlistgetitem_scan_tensorarrayunstack_2_tensorlistfromtensor_01
-scan_while_unsortedsegmentsum_strided_slice_0
scan_while_identity
scan_while_identity_1
scan_while_identity_2
scan_while_identity_3
scan_while_identity_4#
scan_while_scan_strided_slice_1_
[scan_while_tensorarrayv2read_tensorlistgetitem_scan_tensorarrayunstack_tensorlistfromtensorc
_scan_while_tensorarrayv2read_1_tensorlistgetitem_scan_tensorarrayunstack_1_tensorlistfromtensorc
_scan_while_tensorarrayv2read_2_tensorlistgetitem_scan_tensorarrayunstack_2_tensorlistfromtensor/
+scan_while_unsortedsegmentsum_strided_sliceН
<scan/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    `   ┐
.scan/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem]scan_while_tensorarrayv2read_tensorlistgetitem_scan_tensorarrayunstack_tensorlistfromtensor_0scan_while_placeholderEscan/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         `*
element_dtype0П
>scan/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╟
0scan/while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItemascan_while_tensorarrayv2read_1_tensorlistgetitem_scan_tensorarrayunstack_1_tensorlistfromtensor_0scan_while_placeholderGscan/while/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0П
>scan/while/TensorArrayV2Read_2/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╟
0scan/while/TensorArrayV2Read_2/TensorListGetItemTensorListGetItemascan_while_tensorarrayv2read_2_tensorlistgetitem_scan_tensorarrayunstack_2_tensorlistfromtensor_0scan_while_placeholderGscan/while/TensorArrayV2Read_2/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0o
scan/while/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       q
 scan/while/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       q
 scan/while/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ┴
scan/while/strided_sliceStridedSlice7scan/while/TensorArrayV2Read_2/TensorListGetItem:item:0'scan/while/strided_slice/stack:output:0)scan/while/strided_slice/stack_1:output:0)scan/while/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_maskб
scan/while/mulMul5scan/while/TensorArrayV2Read/TensorListGetItem:item:0!scan/while/strided_slice:output:0*
T0*'
_output_shapes
:         `q
 scan/while/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       s
"scan/while/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       s
"scan/while/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ▌
scan/while/strided_slice_1StridedSlice7scan/while/TensorArrayV2Read_1/TensorListGetItem:item:0)scan/while/strided_slice_1/stack:output:0+scan/while/strided_slice_1/stack_1:output:0+scan/while/strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask▌
scan/while/UnsortedSegmentSumUnsortedSegmentSumscan/while/mul:z:0#scan/while/strided_slice_1:output:0-scan_while_unsortedsegmentsum_strided_slice_0*
T0*
Tindices0*'
_output_shapes
:         `▐
/scan/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemscan_while_placeholder_2scan_while_placeholder&scan/while/UnsortedSegmentSum:output:0*
_output_shapes
: *
element_dtype0:щш╥R
scan/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :k
scan/while/addAddV2scan_while_placeholderscan/while/add/y:output:0*
T0*
_output_shapes
: T
scan/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :{
scan/while/add_1AddV2"scan_while_scan_while_loop_counterscan/while/add_1/y:output:0*
T0*
_output_shapes
: V
scan/while/IdentityIdentityscan/while/add_1:z:0*
T0*
_output_shapes
: a
scan/while/Identity_1Identityscan_while_scan_strided_slice*
T0*
_output_shapes
: V
scan/while/Identity_2Identityscan/while/add:z:0*
T0*
_output_shapes
: {
scan/while/Identity_3Identity&scan/while/UnsortedSegmentSum:output:0*
T0*'
_output_shapes
:         `Г
scan/while/Identity_4Identity?scan/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: "3
scan_while_identityscan/while/Identity:output:0"7
scan_while_identity_1scan/while/Identity_1:output:0"7
scan_while_identity_2scan/while/Identity_2:output:0"7
scan_while_identity_3scan/while/Identity_3:output:0"7
scan_while_identity_4scan/while/Identity_4:output:0"D
scan_while_scan_strided_slice_1!scan_while_scan_strided_slice_1_0"─
_scan_while_tensorarrayv2read_1_tensorlistgetitem_scan_tensorarrayunstack_1_tensorlistfromtensorascan_while_tensorarrayv2read_1_tensorlistgetitem_scan_tensorarrayunstack_1_tensorlistfromtensor_0"─
_scan_while_tensorarrayv2read_2_tensorlistgetitem_scan_tensorarrayunstack_2_tensorlistfromtensorascan_while_tensorarrayv2read_2_tensorlistgetitem_scan_tensorarrayunstack_2_tensorlistfromtensor_0"╝
[scan_while_tensorarrayv2read_tensorlistgetitem_scan_tensorarrayunstack_tensorlistfromtensor]scan_while_tensorarrayv2read_tensorlistgetitem_scan_tensorarrayunstack_tensorlistfromtensor_0"\
+scan_while_unsortedsegmentsum_strided_slice-scan_while_unsortedsegmentsum_strided_slice_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : :         `: : : : : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         `:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
Р%
є
P__inference_layer_normalization_2_layer_call_and_return_conditional_losses_19159

inputs+
mul_3_readvariableop_resource:`)
add_readvariableop_resource:`
identityИвadd/ReadVariableOpвmul_3/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/xConst*
_output_shapes
: *
dtype0*
value	B :S
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: _
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
mul_1Mulmul:z:0strided_slice_1:output:0*
T0*
_output_shapes
: _
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_2Mulmul_2/x:output:0strided_slice_2:output:0*
T0*
_output_shapes
: Q
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Н
Reshape/shapePackReshape/shape/0:output:0	mul_1:z:0	mul_2:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:         `L
ones/packedPack	mul_1:z:0*
N*
T0*
_output_shapes
:O

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?e
onesFillones/packed:output:0ones/Const:output:0*
T0*#
_output_shapes
:         M
zeros/packedPack	mul_1:z:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    h
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*#
_output_shapes
:         H
ConstConst*
_output_shapes
: *
dtype0*
valueB J
Const_1Const*
_output_shapes
: *
dtype0*
valueB г
FusedBatchNormV3FusedBatchNormV3Reshape:output:0ones:output:0zeros:output:0Const:output:0Const_1:output:0*
T0*
U0*o
_output_shapes]
[:         `:         :         :         :         :*
data_formatNCHW*
epsilon%oГ:y
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*4
_output_shapes"
 :                  `n
mul_3/ReadVariableOpReadVariableOpmul_3_readvariableop_resource*
_output_shapes
:`*
dtype0}
mul_3MulReshape_1:output:0mul_3/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:`*
dtype0r
addAddV2	mul_3:z:0add/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `c
IdentityIdentityadd:z:0^NoOp*
T0*4
_output_shapes"
 :                  `r
NoOpNoOp^add/ReadVariableOp^mul_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:                  `: : 2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_3/ReadVariableOpmul_3/ReadVariableOp:\ X
4
_output_shapes"
 :                  `
 
_user_specified_nameinputs
Ё9
╧	
mpn_scan_while_body_18595.
*mpn_scan_while_mpn_scan_while_loop_counter)
%mpn_scan_while_mpn_scan_strided_slice
mpn_scan_while_placeholder 
mpn_scan_while_placeholder_1 
mpn_scan_while_placeholder_2-
)mpn_scan_while_mpn_scan_strided_slice_1_0i
empn_scan_while_tensorarrayv2read_tensorlistgetitem_mpn_scan_tensorarrayunstack_tensorlistfromtensor_0m
impn_scan_while_tensorarrayv2read_1_tensorlistgetitem_mpn_scan_tensorarrayunstack_1_tensorlistfromtensor_0m
impn_scan_while_tensorarrayv2read_2_tensorlistgetitem_mpn_scan_tensorarrayunstack_2_tensorlistfromtensor_09
5mpn_scan_while_unsortedsegmentsum_mpn_strided_slice_0
mpn_scan_while_identity
mpn_scan_while_identity_1
mpn_scan_while_identity_2
mpn_scan_while_identity_3
mpn_scan_while_identity_4+
'mpn_scan_while_mpn_scan_strided_slice_1g
cmpn_scan_while_tensorarrayv2read_tensorlistgetitem_mpn_scan_tensorarrayunstack_tensorlistfromtensork
gmpn_scan_while_tensorarrayv2read_1_tensorlistgetitem_mpn_scan_tensorarrayunstack_1_tensorlistfromtensork
gmpn_scan_while_tensorarrayv2read_2_tensorlistgetitem_mpn_scan_tensorarrayunstack_2_tensorlistfromtensor7
3mpn_scan_while_unsortedsegmentsum_mpn_strided_sliceС
@mpn/scan/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    `   ╙
2mpn/scan/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemempn_scan_while_tensorarrayv2read_tensorlistgetitem_mpn_scan_tensorarrayunstack_tensorlistfromtensor_0mpn_scan_while_placeholderImpn/scan/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         `*
element_dtype0У
Bmpn/scan/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       █
4mpn/scan/while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItemimpn_scan_while_tensorarrayv2read_1_tensorlistgetitem_mpn_scan_tensorarrayunstack_1_tensorlistfromtensor_0mpn_scan_while_placeholderKmpn/scan/while/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0У
Bmpn/scan/while/TensorArrayV2Read_2/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       █
4mpn/scan/while/TensorArrayV2Read_2/TensorListGetItemTensorListGetItemimpn_scan_while_tensorarrayv2read_2_tensorlistgetitem_mpn_scan_tensorarrayunstack_2_tensorlistfromtensor_0mpn_scan_while_placeholderKmpn/scan/while/TensorArrayV2Read_2/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0s
"mpn/scan/while/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       u
$mpn/scan/while/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       u
$mpn/scan/while/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╒
mpn/scan/while/strided_sliceStridedSlice;mpn/scan/while/TensorArrayV2Read_2/TensorListGetItem:item:0+mpn/scan/while/strided_slice/stack:output:0-mpn/scan/while/strided_slice/stack_1:output:0-mpn/scan/while/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_maskн
mpn/scan/while/mulMul9mpn/scan/while/TensorArrayV2Read/TensorListGetItem:item:0%mpn/scan/while/strided_slice:output:0*
T0*'
_output_shapes
:         `u
$mpn/scan/while/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       w
&mpn/scan/while/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       w
&mpn/scan/while/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ё
mpn/scan/while/strided_slice_1StridedSlice;mpn/scan/while/TensorArrayV2Read_1/TensorListGetItem:item:0-mpn/scan/while/strided_slice_1/stack:output:0/mpn/scan/while/strided_slice_1/stack_1:output:0/mpn/scan/while/strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_maskё
!mpn/scan/while/UnsortedSegmentSumUnsortedSegmentSummpn/scan/while/mul:z:0'mpn/scan/while/strided_slice_1:output:05mpn_scan_while_unsortedsegmentsum_mpn_strided_slice_0*
T0*
Tindices0*'
_output_shapes
:         `ю
3mpn/scan/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemmpn_scan_while_placeholder_2mpn_scan_while_placeholder*mpn/scan/while/UnsortedSegmentSum:output:0*
_output_shapes
: *
element_dtype0:щш╥V
mpn/scan/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :w
mpn/scan/while/addAddV2mpn_scan_while_placeholdermpn/scan/while/add/y:output:0*
T0*
_output_shapes
: X
mpn/scan/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Л
mpn/scan/while/add_1AddV2*mpn_scan_while_mpn_scan_while_loop_countermpn/scan/while/add_1/y:output:0*
T0*
_output_shapes
: ^
mpn/scan/while/IdentityIdentitympn/scan/while/add_1:z:0*
T0*
_output_shapes
: m
mpn/scan/while/Identity_1Identity%mpn_scan_while_mpn_scan_strided_slice*
T0*
_output_shapes
: ^
mpn/scan/while/Identity_2Identitympn/scan/while/add:z:0*
T0*
_output_shapes
: Г
mpn/scan/while/Identity_3Identity*mpn/scan/while/UnsortedSegmentSum:output:0*
T0*'
_output_shapes
:         `Л
mpn/scan/while/Identity_4IdentityCmpn/scan/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: ";
mpn_scan_while_identity mpn/scan/while/Identity:output:0"?
mpn_scan_while_identity_1"mpn/scan/while/Identity_1:output:0"?
mpn_scan_while_identity_2"mpn/scan/while/Identity_2:output:0"?
mpn_scan_while_identity_3"mpn/scan/while/Identity_3:output:0"?
mpn_scan_while_identity_4"mpn/scan/while/Identity_4:output:0"T
'mpn_scan_while_mpn_scan_strided_slice_1)mpn_scan_while_mpn_scan_strided_slice_1_0"╘
gmpn_scan_while_tensorarrayv2read_1_tensorlistgetitem_mpn_scan_tensorarrayunstack_1_tensorlistfromtensorimpn_scan_while_tensorarrayv2read_1_tensorlistgetitem_mpn_scan_tensorarrayunstack_1_tensorlistfromtensor_0"╘
gmpn_scan_while_tensorarrayv2read_2_tensorlistgetitem_mpn_scan_tensorarrayunstack_2_tensorlistfromtensorimpn_scan_while_tensorarrayv2read_2_tensorlistgetitem_mpn_scan_tensorarrayunstack_2_tensorlistfromtensor_0"╠
cmpn_scan_while_tensorarrayv2read_tensorlistgetitem_mpn_scan_tensorarrayunstack_tensorlistfromtensorempn_scan_while_tensorarrayv2read_tensorlistgetitem_mpn_scan_tensorarrayunstack_tensorlistfromtensor_0"l
3mpn_scan_while_unsortedsegmentsum_mpn_strided_slice5mpn_scan_while_unsortedsegmentsum_mpn_strided_slice_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : :         `: : : : : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         `:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
╦
B
&__inference_lambda_layer_call_fn_18811

inputs
identity╣
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  `* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_lambda_layer_call_and_return_conditional_losses_15364m
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :                  `"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :                  `:\ X
4
_output_shapes"
 :                  `
 
_user_specified_nameinputs
╔
ь
G__inference_sequential_1_layer_call_and_return_conditional_losses_15658

inputs)
layer_normalization_5_15652:`)
layer_normalization_5_15654:`
identityИв-layer_normalization_5/StatefulPartitionedCall└
lambda/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  `* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_lambda_layer_call_and_return_conditional_losses_15461╟
-layer_normalization_5/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0layer_normalization_5_15652layer_normalization_5_15654*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  `*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_layer_normalization_5_layer_call_and_return_conditional_losses_15613Т
IdentityIdentity6layer_normalization_5/StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  `v
NoOpNoOp.^layer_normalization_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:                  `: : 2^
-layer_normalization_5/StatefulPartitionedCall-layer_normalization_5/StatefulPartitionedCall:\ X
4
_output_shapes"
 :                  `
 
_user_specified_nameinputs
Ъ
┼
E__inference_sequential_layer_call_and_return_conditional_losses_15420

inputs
dense_15347:	а`
dense_15349:`)
layer_normalization_4_15414:`)
layer_normalization_4_15416:`
identityИвdense/StatefulPartitionedCallв-layer_normalization_4/StatefulPartitionedCallю
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_15347dense_15349*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  `*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_15346р
lambda/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  `* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_lambda_layer_call_and_return_conditional_losses_15364╟
-layer_normalization_4/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0layer_normalization_4_15414layer_normalization_4_15416*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  `*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_layer_normalization_4_layer_call_and_return_conditional_losses_15413Т
IdentityIdentity6layer_normalization_4/StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  `Ц
NoOpNoOp^dense/StatefulPartitionedCall.^layer_normalization_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):                  а: : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2^
-layer_normalization_4/StatefulPartitionedCall-layer_normalization_4/StatefulPartitionedCall:] Y
5
_output_shapes#
!:                  а
 
_user_specified_nameinputs
о
╗
%__inference_model_layer_call_fn_16468
input_1
input_2
input_3
input_4
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@`
	unknown_6:`
	unknown_7:@
	unknown_8:@
	unknown_9:@`

unknown_10:`

unknown_11:`

unknown_12:`

unknown_13:`

unknown_14:`

unknown_15:	а`

unknown_16:`

unknown_17:`

unknown_18:`

unknown_19:	└`

unknown_20:`

unknown_21:`

unknown_22:`

unknown_23:`

unknown_24:
identityИвStatefulPartitionedCall╚
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2input_3input_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  *<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_16413|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*╔
_input_shapes╖
┤:                  :                  :                  :                  : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
4
_output_shapes"
 :                  
!
_user_specified_name	input_1:]Y
4
_output_shapes"
 :                  
!
_user_specified_name	input_2:]Y
4
_output_shapes"
 :                  
!
_user_specified_name	input_3:]Y
4
_output_shapes"
 :                  
!
_user_specified_name	input_4
уS
Ы
@__inference_model_layer_call_and_return_conditional_losses_17064

inputs
inputs_1
inputs_2
inputs_3!
node_ide1_16986:@
node_ide1_16988:@'
layer_normalization_16992:@'
layer_normalization_16994:@!
edge_ide1_16997:@
edge_ide1_16999:@!
node_ide2_17002:@`
node_ide2_17004:`)
layer_normalization_1_17009:@)
layer_normalization_1_17011:@!
edge_ide2_17014:@`
edge_ide2_17016:`)
layer_normalization_2_17024:`)
layer_normalization_2_17026:`)
layer_normalization_3_17029:`)
layer_normalization_3_17031:`
	mpn_17037:	а`
	mpn_17039:`
	mpn_17041:`
	mpn_17043:`
	mpn_17045:	└`
	mpn_17047:`
	mpn_17049:`
	mpn_17051:`'
node_prediction_17058:`#
node_prediction_17060:
identityИв!edge_ide1/StatefulPartitionedCallв!edge_ide2/StatefulPartitionedCallв+layer_normalization/StatefulPartitionedCallв-layer_normalization_1/StatefulPartitionedCallв-layer_normalization_2/StatefulPartitionedCallв-layer_normalization_3/StatefulPartitionedCallвmpn/StatefulPartitionedCallв!node_ide1/StatefulPartitionedCallв!node_ide2/StatefulPartitionedCallв'node_prediction/StatefulPartitionedCall■
!node_ide1/StatefulPartitionedCallStatefulPartitionedCallinputsnode_ide1_16986node_ide1_16988*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_node_ide1_layer_call_and_return_conditional_losses_15737ф
lambda/PartitionedCallPartitionedCall*node_ide1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_lambda_layer_call_and_return_conditional_losses_16872┐
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0layer_normalization_16992layer_normalization_16994*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *W
fRRP
N__inference_layer_normalization_layer_call_and_return_conditional_losses_15803А
!edge_ide1/StatefulPartitionedCallStatefulPartitionedCallinputs_1edge_ide1_16997edge_ide1_16999*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_edge_ide1_layer_call_and_return_conditional_losses_15839м
!node_ide2/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0node_ide2_17002node_ide2_17004*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  `*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_node_ide2_layer_call_and_return_conditional_losses_15875ц
lambda/PartitionedCall_1PartitionedCall*edge_ide1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_lambda_layer_call_and_return_conditional_losses_16872ц
lambda/PartitionedCall_2PartitionedCall*node_ide2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  `* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_lambda_layer_call_and_return_conditional_losses_15461╔
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCall!lambda/PartitionedCall_1:output:0layer_normalization_1_17009layer_normalization_1_17011*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_layer_normalization_1_layer_call_and_return_conditional_losses_15930о
!edge_ide2/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_1/StatefulPartitionedCall:output:0edge_ide2_17014edge_ide2_17016*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  `*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_edge_ide2_layer_call_and_return_conditional_losses_15966}
,tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
.tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
.tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╩
&tf.__operators__.getitem/strided_sliceStridedSliceinputs_15tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*4
_output_shapes"
 :                  *
ellipsis_maskц
lambda/PartitionedCall_3PartitionedCall*edge_ide2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  `* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_lambda_layer_call_and_return_conditional_losses_15461╔
-layer_normalization_2/StatefulPartitionedCallStatefulPartitionedCall!lambda/PartitionedCall_2:output:0layer_normalization_2_17024layer_normalization_2_17026*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  `*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_layer_normalization_2_layer_call_and_return_conditional_losses_16024╔
-layer_normalization_3/StatefulPartitionedCallStatefulPartitionedCall!lambda/PartitionedCall_3:output:0layer_normalization_3_17029layer_normalization_3_17031*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  `*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_layer_normalization_3_layer_call_and_return_conditional_losses_16077{
tf.ones_like/ones_like/ShapeShape/tf.__operators__.getitem/strided_slice:output:0*
T0*
_output_shapes
:a
tf.ones_like/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?л
tf.ones_like/ones_likeFill%tf.ones_like/ones_like/Shape:output:0%tf.ones_like/ones_like/Const:output:0*
T0*4
_output_shapes"
 :                  ▄
mpn/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_2/StatefulPartitionedCall:output:06layer_normalization_3/StatefulPartitionedCall:output:0inputs_2tf.ones_like/ones_like:output:0inputs_3	mpn_17037	mpn_17039	mpn_17041	mpn_17043	mpn_17045	mpn_17047	mpn_17049	mpn_17051*
Tin
2*
Tout	
2*
_collective_manager_ids
 *╢
_output_shapesг
а:                  `:                  `:                  :                  :                  **
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8В *G
fBR@
>__inference_mpn_layer_call_and_return_conditional_losses_16786┤
'node_prediction/StatefulPartitionedCallStatefulPartitionedCall$mpn/StatefulPartitionedCall:output:0node_prediction_17058node_prediction_17060*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_node_prediction_layer_call_and_return_conditional_losses_16406М
IdentityIdentity0node_prediction/StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  ▄
NoOpNoOp"^edge_ide1/StatefulPartitionedCall"^edge_ide2/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall.^layer_normalization_1/StatefulPartitionedCall.^layer_normalization_2/StatefulPartitionedCall.^layer_normalization_3/StatefulPartitionedCall^mpn/StatefulPartitionedCall"^node_ide1/StatefulPartitionedCall"^node_ide2/StatefulPartitionedCall(^node_prediction/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*╔
_input_shapes╖
┤:                  :                  :                  :                  : : : : : : : : : : : : : : : : : : : : : : : : : : 2F
!edge_ide1/StatefulPartitionedCall!edge_ide1/StatefulPartitionedCall2F
!edge_ide2/StatefulPartitionedCall!edge_ide2/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2^
-layer_normalization_1/StatefulPartitionedCall-layer_normalization_1/StatefulPartitionedCall2^
-layer_normalization_2/StatefulPartitionedCall-layer_normalization_2/StatefulPartitionedCall2^
-layer_normalization_3/StatefulPartitionedCall-layer_normalization_3/StatefulPartitionedCall2:
mpn/StatefulPartitionedCallmpn/StatefulPartitionedCall2F
!node_ide1/StatefulPartitionedCall!node_ide1/StatefulPartitionedCall2F
!node_ide2/StatefulPartitionedCall!node_ide2/StatefulPartitionedCall2R
'node_prediction/StatefulPartitionedCall'node_prediction/StatefulPartitionedCall:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs:\X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs:\X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs:\X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
й╤
═
 __inference__wrapped_model_15309
input_1
input_2
input_3
input_4C
1model_node_ide1_tensordot_readvariableop_resource:@=
/model_node_ide1_biasadd_readvariableop_resource:@E
7model_layer_normalization_mul_3_readvariableop_resource:@C
5model_layer_normalization_add_readvariableop_resource:@C
1model_edge_ide1_tensordot_readvariableop_resource:@=
/model_edge_ide1_biasadd_readvariableop_resource:@C
1model_node_ide2_tensordot_readvariableop_resource:@`=
/model_node_ide2_biasadd_readvariableop_resource:`G
9model_layer_normalization_1_mul_3_readvariableop_resource:@E
7model_layer_normalization_1_add_readvariableop_resource:@C
1model_edge_ide2_tensordot_readvariableop_resource:@`=
/model_edge_ide2_biasadd_readvariableop_resource:`G
9model_layer_normalization_2_mul_3_readvariableop_resource:`E
7model_layer_normalization_2_add_readvariableop_resource:`G
9model_layer_normalization_3_mul_3_readvariableop_resource:`E
7model_layer_normalization_3_add_readvariableop_resource:`O
<model_mpn_sequential_dense_tensordot_readvariableop_resource:	а`H
:model_mpn_sequential_dense_biasadd_readvariableop_resource:`V
Hmodel_mpn_sequential_layer_normalization_4_mul_3_readvariableop_resource:`T
Fmodel_mpn_sequential_layer_normalization_4_add_readvariableop_resource:`F
3model_mpn_dense_1_tensordot_readvariableop_resource:	└`?
1model_mpn_dense_1_biasadd_readvariableop_resource:`X
Jmodel_mpn_sequential_1_layer_normalization_5_mul_3_readvariableop_resource:`V
Hmodel_mpn_sequential_1_layer_normalization_5_add_readvariableop_resource:`I
7model_node_prediction_tensordot_readvariableop_resource:`C
5model_node_prediction_biasadd_readvariableop_resource:
identityИв&model/edge_ide1/BiasAdd/ReadVariableOpв(model/edge_ide1/Tensordot/ReadVariableOpв&model/edge_ide2/BiasAdd/ReadVariableOpв(model/edge_ide2/Tensordot/ReadVariableOpв,model/layer_normalization/add/ReadVariableOpв.model/layer_normalization/mul_3/ReadVariableOpв.model/layer_normalization_1/add/ReadVariableOpв0model/layer_normalization_1/mul_3/ReadVariableOpв.model/layer_normalization_2/add/ReadVariableOpв0model/layer_normalization_2/mul_3/ReadVariableOpв.model/layer_normalization_3/add/ReadVariableOpв0model/layer_normalization_3/mul_3/ReadVariableOpв(model/mpn/dense_1/BiasAdd/ReadVariableOpв*model/mpn/dense_1/Tensordot/ReadVariableOpв1model/mpn/sequential/dense/BiasAdd/ReadVariableOpв3model/mpn/sequential/dense/Tensordot/ReadVariableOpв=model/mpn/sequential/layer_normalization_4/add/ReadVariableOpв?model/mpn/sequential/layer_normalization_4/mul_3/ReadVariableOpв?model/mpn/sequential_1/layer_normalization_5/add/ReadVariableOpвAmodel/mpn/sequential_1/layer_normalization_5/mul_3/ReadVariableOpв&model/node_ide1/BiasAdd/ReadVariableOpв(model/node_ide1/Tensordot/ReadVariableOpв&model/node_ide2/BiasAdd/ReadVariableOpв(model/node_ide2/Tensordot/ReadVariableOpв,model/node_prediction/BiasAdd/ReadVariableOpв.model/node_prediction/Tensordot/ReadVariableOpЪ
(model/node_ide1/Tensordot/ReadVariableOpReadVariableOp1model_node_ide1_tensordot_readvariableop_resource*
_output_shapes

:@*
dtype0h
model/node_ide1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:o
model/node_ide1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       V
model/node_ide1/Tensordot/ShapeShapeinput_1*
T0*
_output_shapes
:i
'model/node_ide1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : √
"model/node_ide1/Tensordot/GatherV2GatherV2(model/node_ide1/Tensordot/Shape:output:0'model/node_ide1/Tensordot/free:output:00model/node_ide1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:k
)model/node_ide1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B :  
$model/node_ide1/Tensordot/GatherV2_1GatherV2(model/node_ide1/Tensordot/Shape:output:0'model/node_ide1/Tensordot/axes:output:02model/node_ide1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:i
model/node_ide1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: Ю
model/node_ide1/Tensordot/ProdProd+model/node_ide1/Tensordot/GatherV2:output:0(model/node_ide1/Tensordot/Const:output:0*
T0*
_output_shapes
: k
!model/node_ide1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: д
 model/node_ide1/Tensordot/Prod_1Prod-model/node_ide1/Tensordot/GatherV2_1:output:0*model/node_ide1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: g
%model/node_ide1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ▄
 model/node_ide1/Tensordot/concatConcatV2'model/node_ide1/Tensordot/free:output:0'model/node_ide1/Tensordot/axes:output:0.model/node_ide1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:й
model/node_ide1/Tensordot/stackPack'model/node_ide1/Tensordot/Prod:output:0)model/node_ide1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:г
#model/node_ide1/Tensordot/transpose	Transposeinput_1)model/node_ide1/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :                  ║
!model/node_ide1/Tensordot/ReshapeReshape'model/node_ide1/Tensordot/transpose:y:0(model/node_ide1/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  ║
 model/node_ide1/Tensordot/MatMulMatMul*model/node_ide1/Tensordot/Reshape:output:00model/node_ide1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @k
!model/node_ide1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@i
'model/node_ide1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ч
"model/node_ide1/Tensordot/concat_1ConcatV2+model/node_ide1/Tensordot/GatherV2:output:0*model/node_ide1/Tensordot/Const_2:output:00model/node_ide1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:╝
model/node_ide1/TensordotReshape*model/node_ide1/Tensordot/MatMul:product:0+model/node_ide1/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  @Т
&model/node_ide1/BiasAdd/ReadVariableOpReadVariableOp/model_node_ide1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0╡
model/node_ide1/BiasAddBiasAdd"model/node_ide1/Tensordot:output:0.model/node_ide1/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  @\
model/lambda/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Я
model/lambda/Gelu/mulMul model/lambda/Gelu/mul/x:output:0 model/node_ide1/BiasAdd:output:0*
T0*4
_output_shapes"
 :                  @]
model/lambda/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *є╡?и
model/lambda/Gelu/truedivRealDiv model/node_ide1/BiasAdd:output:0!model/lambda/Gelu/Cast/x:output:0*
T0*4
_output_shapes"
 :                  @z
model/lambda/Gelu/ErfErfmodel/lambda/Gelu/truediv:z:0*
T0*4
_output_shapes"
 :                  @\
model/lambda/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ъ
model/lambda/Gelu/addAddV2 model/lambda/Gelu/add/x:output:0model/lambda/Gelu/Erf:y:0*
T0*4
_output_shapes"
 :                  @У
model/lambda/Gelu/mul_1Mulmodel/lambda/Gelu/mul:z:0model/lambda/Gelu/add:z:0*
T0*4
_output_shapes"
 :                  @j
model/layer_normalization/ShapeShapemodel/lambda/Gelu/mul_1:z:0*
T0*
_output_shapes
:w
-model/layer_normalization/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/model/layer_normalization/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/model/layer_normalization/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╙
'model/layer_normalization/strided_sliceStridedSlice(model/layer_normalization/Shape:output:06model/layer_normalization/strided_slice/stack:output:08model/layer_normalization/strided_slice/stack_1:output:08model/layer_normalization/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
model/layer_normalization/mul/xConst*
_output_shapes
: *
dtype0*
value	B :б
model/layer_normalization/mulMul(model/layer_normalization/mul/x:output:00model/layer_normalization/strided_slice:output:0*
T0*
_output_shapes
: y
/model/layer_normalization/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:{
1model/layer_normalization/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1model/layer_normalization/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
)model/layer_normalization/strided_slice_1StridedSlice(model/layer_normalization/Shape:output:08model/layer_normalization/strided_slice_1/stack:output:0:model/layer_normalization/strided_slice_1/stack_1:output:0:model/layer_normalization/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЮ
model/layer_normalization/mul_1Mul!model/layer_normalization/mul:z:02model/layer_normalization/strided_slice_1:output:0*
T0*
_output_shapes
: y
/model/layer_normalization/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:{
1model/layer_normalization/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1model/layer_normalization/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
)model/layer_normalization/strided_slice_2StridedSlice(model/layer_normalization/Shape:output:08model/layer_normalization/strided_slice_2/stack:output:0:model/layer_normalization/strided_slice_2/stack_1:output:0:model/layer_normalization/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!model/layer_normalization/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :з
model/layer_normalization/mul_2Mul*model/layer_normalization/mul_2/x:output:02model/layer_normalization/strided_slice_2:output:0*
T0*
_output_shapes
: k
)model/layer_normalization/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :k
)model/layer_normalization/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :П
'model/layer_normalization/Reshape/shapePack2model/layer_normalization/Reshape/shape/0:output:0#model/layer_normalization/mul_1:z:0#model/layer_normalization/mul_2:z:02model/layer_normalization/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:╡
!model/layer_normalization/ReshapeReshapemodel/lambda/Gelu/mul_1:z:00model/layer_normalization/Reshape/shape:output:0*
T0*/
_output_shapes
:         @А
%model/layer_normalization/ones/packedPack#model/layer_normalization/mul_1:z:0*
N*
T0*
_output_shapes
:i
$model/layer_normalization/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?│
model/layer_normalization/onesFill.model/layer_normalization/ones/packed:output:0-model/layer_normalization/ones/Const:output:0*
T0*#
_output_shapes
:         Б
&model/layer_normalization/zeros/packedPack#model/layer_normalization/mul_1:z:0*
N*
T0*
_output_shapes
:j
%model/layer_normalization/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ╢
model/layer_normalization/zerosFill/model/layer_normalization/zeros/packed:output:0.model/layer_normalization/zeros/Const:output:0*
T0*#
_output_shapes
:         b
model/layer_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB d
!model/layer_normalization/Const_1Const*
_output_shapes
: *
dtype0*
valueB ┐
*model/layer_normalization/FusedBatchNormV3FusedBatchNormV3*model/layer_normalization/Reshape:output:0'model/layer_normalization/ones:output:0(model/layer_normalization/zeros:output:0(model/layer_normalization/Const:output:0*model/layer_normalization/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:         @:         :         :         :         :*
data_formatNCHW*
epsilon%oГ:╟
#model/layer_normalization/Reshape_1Reshape.model/layer_normalization/FusedBatchNormV3:y:0(model/layer_normalization/Shape:output:0*
T0*4
_output_shapes"
 :                  @в
.model/layer_normalization/mul_3/ReadVariableOpReadVariableOp7model_layer_normalization_mul_3_readvariableop_resource*
_output_shapes
:@*
dtype0╦
model/layer_normalization/mul_3Mul,model/layer_normalization/Reshape_1:output:06model/layer_normalization/mul_3/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  @Ю
,model/layer_normalization/add/ReadVariableOpReadVariableOp5model_layer_normalization_add_readvariableop_resource*
_output_shapes
:@*
dtype0└
model/layer_normalization/addAddV2#model/layer_normalization/mul_3:z:04model/layer_normalization/add/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  @Ъ
(model/edge_ide1/Tensordot/ReadVariableOpReadVariableOp1model_edge_ide1_tensordot_readvariableop_resource*
_output_shapes

:@*
dtype0h
model/edge_ide1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:o
model/edge_ide1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       V
model/edge_ide1/Tensordot/ShapeShapeinput_2*
T0*
_output_shapes
:i
'model/edge_ide1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : √
"model/edge_ide1/Tensordot/GatherV2GatherV2(model/edge_ide1/Tensordot/Shape:output:0'model/edge_ide1/Tensordot/free:output:00model/edge_ide1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:k
)model/edge_ide1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B :  
$model/edge_ide1/Tensordot/GatherV2_1GatherV2(model/edge_ide1/Tensordot/Shape:output:0'model/edge_ide1/Tensordot/axes:output:02model/edge_ide1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:i
model/edge_ide1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: Ю
model/edge_ide1/Tensordot/ProdProd+model/edge_ide1/Tensordot/GatherV2:output:0(model/edge_ide1/Tensordot/Const:output:0*
T0*
_output_shapes
: k
!model/edge_ide1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: д
 model/edge_ide1/Tensordot/Prod_1Prod-model/edge_ide1/Tensordot/GatherV2_1:output:0*model/edge_ide1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: g
%model/edge_ide1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ▄
 model/edge_ide1/Tensordot/concatConcatV2'model/edge_ide1/Tensordot/free:output:0'model/edge_ide1/Tensordot/axes:output:0.model/edge_ide1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:й
model/edge_ide1/Tensordot/stackPack'model/edge_ide1/Tensordot/Prod:output:0)model/edge_ide1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:г
#model/edge_ide1/Tensordot/transpose	Transposeinput_2)model/edge_ide1/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :                  ║
!model/edge_ide1/Tensordot/ReshapeReshape'model/edge_ide1/Tensordot/transpose:y:0(model/edge_ide1/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  ║
 model/edge_ide1/Tensordot/MatMulMatMul*model/edge_ide1/Tensordot/Reshape:output:00model/edge_ide1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @k
!model/edge_ide1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@i
'model/edge_ide1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ч
"model/edge_ide1/Tensordot/concat_1ConcatV2+model/edge_ide1/Tensordot/GatherV2:output:0*model/edge_ide1/Tensordot/Const_2:output:00model/edge_ide1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:╝
model/edge_ide1/TensordotReshape*model/edge_ide1/Tensordot/MatMul:product:0+model/edge_ide1/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  @Т
&model/edge_ide1/BiasAdd/ReadVariableOpReadVariableOp/model_edge_ide1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0╡
model/edge_ide1/BiasAddBiasAdd"model/edge_ide1/Tensordot:output:0.model/edge_ide1/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  @Ъ
(model/node_ide2/Tensordot/ReadVariableOpReadVariableOp1model_node_ide2_tensordot_readvariableop_resource*
_output_shapes

:@`*
dtype0h
model/node_ide2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:o
model/node_ide2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       p
model/node_ide2/Tensordot/ShapeShape!model/layer_normalization/add:z:0*
T0*
_output_shapes
:i
'model/node_ide2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : √
"model/node_ide2/Tensordot/GatherV2GatherV2(model/node_ide2/Tensordot/Shape:output:0'model/node_ide2/Tensordot/free:output:00model/node_ide2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:k
)model/node_ide2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B :  
$model/node_ide2/Tensordot/GatherV2_1GatherV2(model/node_ide2/Tensordot/Shape:output:0'model/node_ide2/Tensordot/axes:output:02model/node_ide2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:i
model/node_ide2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: Ю
model/node_ide2/Tensordot/ProdProd+model/node_ide2/Tensordot/GatherV2:output:0(model/node_ide2/Tensordot/Const:output:0*
T0*
_output_shapes
: k
!model/node_ide2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: д
 model/node_ide2/Tensordot/Prod_1Prod-model/node_ide2/Tensordot/GatherV2_1:output:0*model/node_ide2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: g
%model/node_ide2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ▄
 model/node_ide2/Tensordot/concatConcatV2'model/node_ide2/Tensordot/free:output:0'model/node_ide2/Tensordot/axes:output:0.model/node_ide2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:й
model/node_ide2/Tensordot/stackPack'model/node_ide2/Tensordot/Prod:output:0)model/node_ide2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:╜
#model/node_ide2/Tensordot/transpose	Transpose!model/layer_normalization/add:z:0)model/node_ide2/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :                  @║
!model/node_ide2/Tensordot/ReshapeReshape'model/node_ide2/Tensordot/transpose:y:0(model/node_ide2/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  ║
 model/node_ide2/Tensordot/MatMulMatMul*model/node_ide2/Tensordot/Reshape:output:00model/node_ide2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `k
!model/node_ide2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:`i
'model/node_ide2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ч
"model/node_ide2/Tensordot/concat_1ConcatV2+model/node_ide2/Tensordot/GatherV2:output:0*model/node_ide2/Tensordot/Const_2:output:00model/node_ide2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:╝
model/node_ide2/TensordotReshape*model/node_ide2/Tensordot/MatMul:product:0+model/node_ide2/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  `Т
&model/node_ide2/BiasAdd/ReadVariableOpReadVariableOp/model_node_ide2_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0╡
model/node_ide2/BiasAddBiasAdd"model/node_ide2/Tensordot:output:0.model/node_ide2/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `^
model/lambda/Gelu_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?г
model/lambda/Gelu_1/mulMul"model/lambda/Gelu_1/mul/x:output:0 model/edge_ide1/BiasAdd:output:0*
T0*4
_output_shapes"
 :                  @_
model/lambda/Gelu_1/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *є╡?м
model/lambda/Gelu_1/truedivRealDiv model/edge_ide1/BiasAdd:output:0#model/lambda/Gelu_1/Cast/x:output:0*
T0*4
_output_shapes"
 :                  @~
model/lambda/Gelu_1/ErfErfmodel/lambda/Gelu_1/truediv:z:0*
T0*4
_output_shapes"
 :                  @^
model/lambda/Gelu_1/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?а
model/lambda/Gelu_1/addAddV2"model/lambda/Gelu_1/add/x:output:0model/lambda/Gelu_1/Erf:y:0*
T0*4
_output_shapes"
 :                  @Щ
model/lambda/Gelu_1/mul_1Mulmodel/lambda/Gelu_1/mul:z:0model/lambda/Gelu_1/add:z:0*
T0*4
_output_shapes"
 :                  @^
model/lambda/Gelu_2/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?г
model/lambda/Gelu_2/mulMul"model/lambda/Gelu_2/mul/x:output:0 model/node_ide2/BiasAdd:output:0*
T0*4
_output_shapes"
 :                  `_
model/lambda/Gelu_2/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *є╡?м
model/lambda/Gelu_2/truedivRealDiv model/node_ide2/BiasAdd:output:0#model/lambda/Gelu_2/Cast/x:output:0*
T0*4
_output_shapes"
 :                  `~
model/lambda/Gelu_2/ErfErfmodel/lambda/Gelu_2/truediv:z:0*
T0*4
_output_shapes"
 :                  `^
model/lambda/Gelu_2/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?а
model/lambda/Gelu_2/addAddV2"model/lambda/Gelu_2/add/x:output:0model/lambda/Gelu_2/Erf:y:0*
T0*4
_output_shapes"
 :                  `Щ
model/lambda/Gelu_2/mul_1Mulmodel/lambda/Gelu_2/mul:z:0model/lambda/Gelu_2/add:z:0*
T0*4
_output_shapes"
 :                  `n
!model/layer_normalization_1/ShapeShapemodel/lambda/Gelu_1/mul_1:z:0*
T0*
_output_shapes
:y
/model/layer_normalization_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1model/layer_normalization_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1model/layer_normalization_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:▌
)model/layer_normalization_1/strided_sliceStridedSlice*model/layer_normalization_1/Shape:output:08model/layer_normalization_1/strided_slice/stack:output:0:model/layer_normalization_1/strided_slice/stack_1:output:0:model/layer_normalization_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!model/layer_normalization_1/mul/xConst*
_output_shapes
: *
dtype0*
value	B :з
model/layer_normalization_1/mulMul*model/layer_normalization_1/mul/x:output:02model/layer_normalization_1/strided_slice:output:0*
T0*
_output_shapes
: {
1model/layer_normalization_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3model/layer_normalization_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3model/layer_normalization_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
+model/layer_normalization_1/strided_slice_1StridedSlice*model/layer_normalization_1/Shape:output:0:model/layer_normalization_1/strided_slice_1/stack:output:0<model/layer_normalization_1/strided_slice_1/stack_1:output:0<model/layer_normalization_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskд
!model/layer_normalization_1/mul_1Mul#model/layer_normalization_1/mul:z:04model/layer_normalization_1/strided_slice_1:output:0*
T0*
_output_shapes
: {
1model/layer_normalization_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3model/layer_normalization_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3model/layer_normalization_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
+model/layer_normalization_1/strided_slice_2StridedSlice*model/layer_normalization_1/Shape:output:0:model/layer_normalization_1/strided_slice_2/stack:output:0<model/layer_normalization_1/strided_slice_2/stack_1:output:0<model/layer_normalization_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#model/layer_normalization_1/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :н
!model/layer_normalization_1/mul_2Mul,model/layer_normalization_1/mul_2/x:output:04model/layer_normalization_1/strided_slice_2:output:0*
T0*
_output_shapes
: m
+model/layer_normalization_1/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :m
+model/layer_normalization_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Щ
)model/layer_normalization_1/Reshape/shapePack4model/layer_normalization_1/Reshape/shape/0:output:0%model/layer_normalization_1/mul_1:z:0%model/layer_normalization_1/mul_2:z:04model/layer_normalization_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:╗
#model/layer_normalization_1/ReshapeReshapemodel/lambda/Gelu_1/mul_1:z:02model/layer_normalization_1/Reshape/shape:output:0*
T0*/
_output_shapes
:         @Д
'model/layer_normalization_1/ones/packedPack%model/layer_normalization_1/mul_1:z:0*
N*
T0*
_output_shapes
:k
&model/layer_normalization_1/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╣
 model/layer_normalization_1/onesFill0model/layer_normalization_1/ones/packed:output:0/model/layer_normalization_1/ones/Const:output:0*
T0*#
_output_shapes
:         Е
(model/layer_normalization_1/zeros/packedPack%model/layer_normalization_1/mul_1:z:0*
N*
T0*
_output_shapes
:l
'model/layer_normalization_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ╝
!model/layer_normalization_1/zerosFill1model/layer_normalization_1/zeros/packed:output:00model/layer_normalization_1/zeros/Const:output:0*
T0*#
_output_shapes
:         d
!model/layer_normalization_1/ConstConst*
_output_shapes
: *
dtype0*
valueB f
#model/layer_normalization_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB ╦
,model/layer_normalization_1/FusedBatchNormV3FusedBatchNormV3,model/layer_normalization_1/Reshape:output:0)model/layer_normalization_1/ones:output:0*model/layer_normalization_1/zeros:output:0*model/layer_normalization_1/Const:output:0,model/layer_normalization_1/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:         @:         :         :         :         :*
data_formatNCHW*
epsilon%oГ:═
%model/layer_normalization_1/Reshape_1Reshape0model/layer_normalization_1/FusedBatchNormV3:y:0*model/layer_normalization_1/Shape:output:0*
T0*4
_output_shapes"
 :                  @ж
0model/layer_normalization_1/mul_3/ReadVariableOpReadVariableOp9model_layer_normalization_1_mul_3_readvariableop_resource*
_output_shapes
:@*
dtype0╤
!model/layer_normalization_1/mul_3Mul.model/layer_normalization_1/Reshape_1:output:08model/layer_normalization_1/mul_3/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  @в
.model/layer_normalization_1/add/ReadVariableOpReadVariableOp7model_layer_normalization_1_add_readvariableop_resource*
_output_shapes
:@*
dtype0╞
model/layer_normalization_1/addAddV2%model/layer_normalization_1/mul_3:z:06model/layer_normalization_1/add/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  @Ъ
(model/edge_ide2/Tensordot/ReadVariableOpReadVariableOp1model_edge_ide2_tensordot_readvariableop_resource*
_output_shapes

:@`*
dtype0h
model/edge_ide2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:o
model/edge_ide2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       r
model/edge_ide2/Tensordot/ShapeShape#model/layer_normalization_1/add:z:0*
T0*
_output_shapes
:i
'model/edge_ide2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : √
"model/edge_ide2/Tensordot/GatherV2GatherV2(model/edge_ide2/Tensordot/Shape:output:0'model/edge_ide2/Tensordot/free:output:00model/edge_ide2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:k
)model/edge_ide2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B :  
$model/edge_ide2/Tensordot/GatherV2_1GatherV2(model/edge_ide2/Tensordot/Shape:output:0'model/edge_ide2/Tensordot/axes:output:02model/edge_ide2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:i
model/edge_ide2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: Ю
model/edge_ide2/Tensordot/ProdProd+model/edge_ide2/Tensordot/GatherV2:output:0(model/edge_ide2/Tensordot/Const:output:0*
T0*
_output_shapes
: k
!model/edge_ide2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: д
 model/edge_ide2/Tensordot/Prod_1Prod-model/edge_ide2/Tensordot/GatherV2_1:output:0*model/edge_ide2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: g
%model/edge_ide2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ▄
 model/edge_ide2/Tensordot/concatConcatV2'model/edge_ide2/Tensordot/free:output:0'model/edge_ide2/Tensordot/axes:output:0.model/edge_ide2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:й
model/edge_ide2/Tensordot/stackPack'model/edge_ide2/Tensordot/Prod:output:0)model/edge_ide2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:┐
#model/edge_ide2/Tensordot/transpose	Transpose#model/layer_normalization_1/add:z:0)model/edge_ide2/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :                  @║
!model/edge_ide2/Tensordot/ReshapeReshape'model/edge_ide2/Tensordot/transpose:y:0(model/edge_ide2/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  ║
 model/edge_ide2/Tensordot/MatMulMatMul*model/edge_ide2/Tensordot/Reshape:output:00model/edge_ide2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `k
!model/edge_ide2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:`i
'model/edge_ide2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ч
"model/edge_ide2/Tensordot/concat_1ConcatV2+model/edge_ide2/Tensordot/GatherV2:output:0*model/edge_ide2/Tensordot/Const_2:output:00model/edge_ide2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:╝
model/edge_ide2/TensordotReshape*model/edge_ide2/Tensordot/MatMul:product:0+model/edge_ide2/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  `Т
&model/edge_ide2/BiasAdd/ReadVariableOpReadVariableOp/model_edge_ide2_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0╡
model/edge_ide2/BiasAddBiasAdd"model/edge_ide2/Tensordot:output:0.model/edge_ide2/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `Г
2model/tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        Е
4model/tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       Е
4model/tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      с
,model/tf.__operators__.getitem/strided_sliceStridedSliceinput_2;model/tf.__operators__.getitem/strided_slice/stack:output:0=model/tf.__operators__.getitem/strided_slice/stack_1:output:0=model/tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*4
_output_shapes"
 :                  *
ellipsis_mask^
model/lambda/Gelu_3/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?г
model/lambda/Gelu_3/mulMul"model/lambda/Gelu_3/mul/x:output:0 model/edge_ide2/BiasAdd:output:0*
T0*4
_output_shapes"
 :                  `_
model/lambda/Gelu_3/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *є╡?м
model/lambda/Gelu_3/truedivRealDiv model/edge_ide2/BiasAdd:output:0#model/lambda/Gelu_3/Cast/x:output:0*
T0*4
_output_shapes"
 :                  `~
model/lambda/Gelu_3/ErfErfmodel/lambda/Gelu_3/truediv:z:0*
T0*4
_output_shapes"
 :                  `^
model/lambda/Gelu_3/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?а
model/lambda/Gelu_3/addAddV2"model/lambda/Gelu_3/add/x:output:0model/lambda/Gelu_3/Erf:y:0*
T0*4
_output_shapes"
 :                  `Щ
model/lambda/Gelu_3/mul_1Mulmodel/lambda/Gelu_3/mul:z:0model/lambda/Gelu_3/add:z:0*
T0*4
_output_shapes"
 :                  `n
!model/layer_normalization_2/ShapeShapemodel/lambda/Gelu_2/mul_1:z:0*
T0*
_output_shapes
:y
/model/layer_normalization_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1model/layer_normalization_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1model/layer_normalization_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:▌
)model/layer_normalization_2/strided_sliceStridedSlice*model/layer_normalization_2/Shape:output:08model/layer_normalization_2/strided_slice/stack:output:0:model/layer_normalization_2/strided_slice/stack_1:output:0:model/layer_normalization_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!model/layer_normalization_2/mul/xConst*
_output_shapes
: *
dtype0*
value	B :з
model/layer_normalization_2/mulMul*model/layer_normalization_2/mul/x:output:02model/layer_normalization_2/strided_slice:output:0*
T0*
_output_shapes
: {
1model/layer_normalization_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3model/layer_normalization_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3model/layer_normalization_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
+model/layer_normalization_2/strided_slice_1StridedSlice*model/layer_normalization_2/Shape:output:0:model/layer_normalization_2/strided_slice_1/stack:output:0<model/layer_normalization_2/strided_slice_1/stack_1:output:0<model/layer_normalization_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskд
!model/layer_normalization_2/mul_1Mul#model/layer_normalization_2/mul:z:04model/layer_normalization_2/strided_slice_1:output:0*
T0*
_output_shapes
: {
1model/layer_normalization_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3model/layer_normalization_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3model/layer_normalization_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
+model/layer_normalization_2/strided_slice_2StridedSlice*model/layer_normalization_2/Shape:output:0:model/layer_normalization_2/strided_slice_2/stack:output:0<model/layer_normalization_2/strided_slice_2/stack_1:output:0<model/layer_normalization_2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#model/layer_normalization_2/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :н
!model/layer_normalization_2/mul_2Mul,model/layer_normalization_2/mul_2/x:output:04model/layer_normalization_2/strided_slice_2:output:0*
T0*
_output_shapes
: m
+model/layer_normalization_2/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :m
+model/layer_normalization_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Щ
)model/layer_normalization_2/Reshape/shapePack4model/layer_normalization_2/Reshape/shape/0:output:0%model/layer_normalization_2/mul_1:z:0%model/layer_normalization_2/mul_2:z:04model/layer_normalization_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:╗
#model/layer_normalization_2/ReshapeReshapemodel/lambda/Gelu_2/mul_1:z:02model/layer_normalization_2/Reshape/shape:output:0*
T0*/
_output_shapes
:         `Д
'model/layer_normalization_2/ones/packedPack%model/layer_normalization_2/mul_1:z:0*
N*
T0*
_output_shapes
:k
&model/layer_normalization_2/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╣
 model/layer_normalization_2/onesFill0model/layer_normalization_2/ones/packed:output:0/model/layer_normalization_2/ones/Const:output:0*
T0*#
_output_shapes
:         Е
(model/layer_normalization_2/zeros/packedPack%model/layer_normalization_2/mul_1:z:0*
N*
T0*
_output_shapes
:l
'model/layer_normalization_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ╝
!model/layer_normalization_2/zerosFill1model/layer_normalization_2/zeros/packed:output:00model/layer_normalization_2/zeros/Const:output:0*
T0*#
_output_shapes
:         d
!model/layer_normalization_2/ConstConst*
_output_shapes
: *
dtype0*
valueB f
#model/layer_normalization_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB ╦
,model/layer_normalization_2/FusedBatchNormV3FusedBatchNormV3,model/layer_normalization_2/Reshape:output:0)model/layer_normalization_2/ones:output:0*model/layer_normalization_2/zeros:output:0*model/layer_normalization_2/Const:output:0,model/layer_normalization_2/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:         `:         :         :         :         :*
data_formatNCHW*
epsilon%oГ:═
%model/layer_normalization_2/Reshape_1Reshape0model/layer_normalization_2/FusedBatchNormV3:y:0*model/layer_normalization_2/Shape:output:0*
T0*4
_output_shapes"
 :                  `ж
0model/layer_normalization_2/mul_3/ReadVariableOpReadVariableOp9model_layer_normalization_2_mul_3_readvariableop_resource*
_output_shapes
:`*
dtype0╤
!model/layer_normalization_2/mul_3Mul.model/layer_normalization_2/Reshape_1:output:08model/layer_normalization_2/mul_3/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `в
.model/layer_normalization_2/add/ReadVariableOpReadVariableOp7model_layer_normalization_2_add_readvariableop_resource*
_output_shapes
:`*
dtype0╞
model/layer_normalization_2/addAddV2%model/layer_normalization_2/mul_3:z:06model/layer_normalization_2/add/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `n
!model/layer_normalization_3/ShapeShapemodel/lambda/Gelu_3/mul_1:z:0*
T0*
_output_shapes
:y
/model/layer_normalization_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1model/layer_normalization_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1model/layer_normalization_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:▌
)model/layer_normalization_3/strided_sliceStridedSlice*model/layer_normalization_3/Shape:output:08model/layer_normalization_3/strided_slice/stack:output:0:model/layer_normalization_3/strided_slice/stack_1:output:0:model/layer_normalization_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!model/layer_normalization_3/mul/xConst*
_output_shapes
: *
dtype0*
value	B :з
model/layer_normalization_3/mulMul*model/layer_normalization_3/mul/x:output:02model/layer_normalization_3/strided_slice:output:0*
T0*
_output_shapes
: {
1model/layer_normalization_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3model/layer_normalization_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3model/layer_normalization_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
+model/layer_normalization_3/strided_slice_1StridedSlice*model/layer_normalization_3/Shape:output:0:model/layer_normalization_3/strided_slice_1/stack:output:0<model/layer_normalization_3/strided_slice_1/stack_1:output:0<model/layer_normalization_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskд
!model/layer_normalization_3/mul_1Mul#model/layer_normalization_3/mul:z:04model/layer_normalization_3/strided_slice_1:output:0*
T0*
_output_shapes
: {
1model/layer_normalization_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3model/layer_normalization_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3model/layer_normalization_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
+model/layer_normalization_3/strided_slice_2StridedSlice*model/layer_normalization_3/Shape:output:0:model/layer_normalization_3/strided_slice_2/stack:output:0<model/layer_normalization_3/strided_slice_2/stack_1:output:0<model/layer_normalization_3/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#model/layer_normalization_3/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :н
!model/layer_normalization_3/mul_2Mul,model/layer_normalization_3/mul_2/x:output:04model/layer_normalization_3/strided_slice_2:output:0*
T0*
_output_shapes
: m
+model/layer_normalization_3/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :m
+model/layer_normalization_3/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Щ
)model/layer_normalization_3/Reshape/shapePack4model/layer_normalization_3/Reshape/shape/0:output:0%model/layer_normalization_3/mul_1:z:0%model/layer_normalization_3/mul_2:z:04model/layer_normalization_3/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:╗
#model/layer_normalization_3/ReshapeReshapemodel/lambda/Gelu_3/mul_1:z:02model/layer_normalization_3/Reshape/shape:output:0*
T0*/
_output_shapes
:         `Д
'model/layer_normalization_3/ones/packedPack%model/layer_normalization_3/mul_1:z:0*
N*
T0*
_output_shapes
:k
&model/layer_normalization_3/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╣
 model/layer_normalization_3/onesFill0model/layer_normalization_3/ones/packed:output:0/model/layer_normalization_3/ones/Const:output:0*
T0*#
_output_shapes
:         Е
(model/layer_normalization_3/zeros/packedPack%model/layer_normalization_3/mul_1:z:0*
N*
T0*
_output_shapes
:l
'model/layer_normalization_3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ╝
!model/layer_normalization_3/zerosFill1model/layer_normalization_3/zeros/packed:output:00model/layer_normalization_3/zeros/Const:output:0*
T0*#
_output_shapes
:         d
!model/layer_normalization_3/ConstConst*
_output_shapes
: *
dtype0*
valueB f
#model/layer_normalization_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB ╦
,model/layer_normalization_3/FusedBatchNormV3FusedBatchNormV3,model/layer_normalization_3/Reshape:output:0)model/layer_normalization_3/ones:output:0*model/layer_normalization_3/zeros:output:0*model/layer_normalization_3/Const:output:0,model/layer_normalization_3/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:         `:         :         :         :         :*
data_formatNCHW*
epsilon%oГ:═
%model/layer_normalization_3/Reshape_1Reshape0model/layer_normalization_3/FusedBatchNormV3:y:0*model/layer_normalization_3/Shape:output:0*
T0*4
_output_shapes"
 :                  `ж
0model/layer_normalization_3/mul_3/ReadVariableOpReadVariableOp9model_layer_normalization_3_mul_3_readvariableop_resource*
_output_shapes
:`*
dtype0╤
!model/layer_normalization_3/mul_3Mul.model/layer_normalization_3/Reshape_1:output:08model/layer_normalization_3/mul_3/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `в
.model/layer_normalization_3/add/ReadVariableOpReadVariableOp7model_layer_normalization_3_add_readvariableop_resource*
_output_shapes
:`*
dtype0╞
model/layer_normalization_3/addAddV2%model/layer_normalization_3/mul_3:z:06model/layer_normalization_3/add/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `З
"model/tf.ones_like/ones_like/ShapeShape5model/tf.__operators__.getitem/strided_slice:output:0*
T0*
_output_shapes
:g
"model/tf.ones_like/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╜
model/tf.ones_like/ones_likeFill+model/tf.ones_like/ones_like/Shape:output:0+model/tf.ones_like/ones_like/Const:output:0*
T0*4
_output_shapes"
 :                  b
model/mpn/ShapeShape#model/layer_normalization_2/add:z:0*
T0*
_output_shapes
:g
model/mpn/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:i
model/mpn/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
model/mpn/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Г
model/mpn/strided_sliceStridedSlicemodel/mpn/Shape:output:0&model/mpn/strided_slice/stack:output:0(model/mpn/strided_slice/stack_1:output:0(model/mpn/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskH
model/mpn/Shape_1Shapeinput_3*
T0*
_output_shapes
:i
model/mpn/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:k
!model/mpn/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!model/mpn/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
model/mpn/strided_slice_1StridedSlicemodel/mpn/Shape_1:output:0(model/mpn/strided_slice_1/stack:output:0*model/mpn/strided_slice_1/stack_1:output:0*model/mpn/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
model/mpn/Shape_2Shape#model/layer_normalization_2/add:z:0*
T0*
_output_shapes
:i
model/mpn/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!model/mpn/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!model/mpn/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
model/mpn/strided_slice_2StridedSlicemodel/mpn/Shape_2:output:0(model/mpn/strided_slice_2/stack:output:0*model/mpn/strided_slice_2/stack_1:output:0*model/mpn/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
model/mpn/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :ц
model/mpn/GatherV2GatherV2#model/layer_normalization_2/add:z:0input_3 model/mpn/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*8
_output_shapes&
$:"                  `*

batch_dims\
model/mpn/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :└╣
model/mpn/Reshape/shapePack"model/mpn/strided_slice_2:output:0"model/mpn/strided_slice_1:output:0"model/mpn/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:Ы
model/mpn/ReshapeReshapemodel/mpn/GatherV2:output:0 model/mpn/Reshape/shape:output:0*
T0*5
_output_shapes#
!:                  └`
model/mpn/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
         ╞
model/mpn/concatConcatV2model/mpn/Reshape:output:0#model/layer_normalization_3/add:z:0model/mpn/concat/axis:output:0*
N*
T0*5
_output_shapes#
!:                  а▒
3model/mpn/sequential/dense/Tensordot/ReadVariableOpReadVariableOp<model_mpn_sequential_dense_tensordot_readvariableop_resource*
_output_shapes
:	а`*
dtype0s
)model/mpn/sequential/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:z
)model/mpn/sequential/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       s
*model/mpn/sequential/dense/Tensordot/ShapeShapemodel/mpn/concat:output:0*
T0*
_output_shapes
:t
2model/mpn/sequential/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : з
-model/mpn/sequential/dense/Tensordot/GatherV2GatherV23model/mpn/sequential/dense/Tensordot/Shape:output:02model/mpn/sequential/dense/Tensordot/free:output:0;model/mpn/sequential/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:v
4model/mpn/sequential/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : л
/model/mpn/sequential/dense/Tensordot/GatherV2_1GatherV23model/mpn/sequential/dense/Tensordot/Shape:output:02model/mpn/sequential/dense/Tensordot/axes:output:0=model/mpn/sequential/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:t
*model/mpn/sequential/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ┐
)model/mpn/sequential/dense/Tensordot/ProdProd6model/mpn/sequential/dense/Tensordot/GatherV2:output:03model/mpn/sequential/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: v
,model/mpn/sequential/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ┼
+model/mpn/sequential/dense/Tensordot/Prod_1Prod8model/mpn/sequential/dense/Tensordot/GatherV2_1:output:05model/mpn/sequential/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: r
0model/mpn/sequential/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : И
+model/mpn/sequential/dense/Tensordot/concatConcatV22model/mpn/sequential/dense/Tensordot/free:output:02model/mpn/sequential/dense/Tensordot/axes:output:09model/mpn/sequential/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:╩
*model/mpn/sequential/dense/Tensordot/stackPack2model/mpn/sequential/dense/Tensordot/Prod:output:04model/mpn/sequential/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:╠
.model/mpn/sequential/dense/Tensordot/transpose	Transposemodel/mpn/concat:output:04model/mpn/sequential/dense/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:                  а█
,model/mpn/sequential/dense/Tensordot/ReshapeReshape2model/mpn/sequential/dense/Tensordot/transpose:y:03model/mpn/sequential/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  █
+model/mpn/sequential/dense/Tensordot/MatMulMatMul5model/mpn/sequential/dense/Tensordot/Reshape:output:0;model/mpn/sequential/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `v
,model/mpn/sequential/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:`t
2model/mpn/sequential/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : У
-model/mpn/sequential/dense/Tensordot/concat_1ConcatV26model/mpn/sequential/dense/Tensordot/GatherV2:output:05model/mpn/sequential/dense/Tensordot/Const_2:output:0;model/mpn/sequential/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:▌
$model/mpn/sequential/dense/TensordotReshape5model/mpn/sequential/dense/Tensordot/MatMul:product:06model/mpn/sequential/dense/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  `и
1model/mpn/sequential/dense/BiasAdd/ReadVariableOpReadVariableOp:model_mpn_sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0╓
"model/mpn/sequential/dense/BiasAddBiasAdd-model/mpn/sequential/dense/Tensordot:output:09model/mpn/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `k
&model/mpn/sequential/lambda/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?╚
$model/mpn/sequential/lambda/Gelu/mulMul/model/mpn/sequential/lambda/Gelu/mul/x:output:0+model/mpn/sequential/dense/BiasAdd:output:0*
T0*4
_output_shapes"
 :                  `l
'model/mpn/sequential/lambda/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *є╡?╤
(model/mpn/sequential/lambda/Gelu/truedivRealDiv+model/mpn/sequential/dense/BiasAdd:output:00model/mpn/sequential/lambda/Gelu/Cast/x:output:0*
T0*4
_output_shapes"
 :                  `Ш
$model/mpn/sequential/lambda/Gelu/ErfErf,model/mpn/sequential/lambda/Gelu/truediv:z:0*
T0*4
_output_shapes"
 :                  `k
&model/mpn/sequential/lambda/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╟
$model/mpn/sequential/lambda/Gelu/addAddV2/model/mpn/sequential/lambda/Gelu/add/x:output:0(model/mpn/sequential/lambda/Gelu/Erf:y:0*
T0*4
_output_shapes"
 :                  `└
&model/mpn/sequential/lambda/Gelu/mul_1Mul(model/mpn/sequential/lambda/Gelu/mul:z:0(model/mpn/sequential/lambda/Gelu/add:z:0*
T0*4
_output_shapes"
 :                  `К
0model/mpn/sequential/layer_normalization_4/ShapeShape*model/mpn/sequential/lambda/Gelu/mul_1:z:0*
T0*
_output_shapes
:И
>model/mpn/sequential/layer_normalization_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: К
@model/mpn/sequential/layer_normalization_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:К
@model/mpn/sequential/layer_normalization_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:и
8model/mpn/sequential/layer_normalization_4/strided_sliceStridedSlice9model/mpn/sequential/layer_normalization_4/Shape:output:0Gmodel/mpn/sequential/layer_normalization_4/strided_slice/stack:output:0Imodel/mpn/sequential/layer_normalization_4/strided_slice/stack_1:output:0Imodel/mpn/sequential/layer_normalization_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskr
0model/mpn/sequential/layer_normalization_4/mul/xConst*
_output_shapes
: *
dtype0*
value	B :╘
.model/mpn/sequential/layer_normalization_4/mulMul9model/mpn/sequential/layer_normalization_4/mul/x:output:0Amodel/mpn/sequential/layer_normalization_4/strided_slice:output:0*
T0*
_output_shapes
: К
@model/mpn/sequential/layer_normalization_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:М
Bmodel/mpn/sequential/layer_normalization_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:М
Bmodel/mpn/sequential/layer_normalization_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:░
:model/mpn/sequential/layer_normalization_4/strided_slice_1StridedSlice9model/mpn/sequential/layer_normalization_4/Shape:output:0Imodel/mpn/sequential/layer_normalization_4/strided_slice_1/stack:output:0Kmodel/mpn/sequential/layer_normalization_4/strided_slice_1/stack_1:output:0Kmodel/mpn/sequential/layer_normalization_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask╤
0model/mpn/sequential/layer_normalization_4/mul_1Mul2model/mpn/sequential/layer_normalization_4/mul:z:0Cmodel/mpn/sequential/layer_normalization_4/strided_slice_1:output:0*
T0*
_output_shapes
: К
@model/mpn/sequential/layer_normalization_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:М
Bmodel/mpn/sequential/layer_normalization_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:М
Bmodel/mpn/sequential/layer_normalization_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:░
:model/mpn/sequential/layer_normalization_4/strided_slice_2StridedSlice9model/mpn/sequential/layer_normalization_4/Shape:output:0Imodel/mpn/sequential/layer_normalization_4/strided_slice_2/stack:output:0Kmodel/mpn/sequential/layer_normalization_4/strided_slice_2/stack_1:output:0Kmodel/mpn/sequential/layer_normalization_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
2model/mpn/sequential/layer_normalization_4/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :┌
0model/mpn/sequential/layer_normalization_4/mul_2Mul;model/mpn/sequential/layer_normalization_4/mul_2/x:output:0Cmodel/mpn/sequential/layer_normalization_4/strided_slice_2:output:0*
T0*
_output_shapes
: |
:model/mpn/sequential/layer_normalization_4/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :|
:model/mpn/sequential/layer_normalization_4/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :ф
8model/mpn/sequential/layer_normalization_4/Reshape/shapePackCmodel/mpn/sequential/layer_normalization_4/Reshape/shape/0:output:04model/mpn/sequential/layer_normalization_4/mul_1:z:04model/mpn/sequential/layer_normalization_4/mul_2:z:0Cmodel/mpn/sequential/layer_normalization_4/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:ц
2model/mpn/sequential/layer_normalization_4/ReshapeReshape*model/mpn/sequential/lambda/Gelu/mul_1:z:0Amodel/mpn/sequential/layer_normalization_4/Reshape/shape:output:0*
T0*/
_output_shapes
:         `в
6model/mpn/sequential/layer_normalization_4/ones/packedPack4model/mpn/sequential/layer_normalization_4/mul_1:z:0*
N*
T0*
_output_shapes
:z
5model/mpn/sequential/layer_normalization_4/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?ц
/model/mpn/sequential/layer_normalization_4/onesFill?model/mpn/sequential/layer_normalization_4/ones/packed:output:0>model/mpn/sequential/layer_normalization_4/ones/Const:output:0*
T0*#
_output_shapes
:         г
7model/mpn/sequential/layer_normalization_4/zeros/packedPack4model/mpn/sequential/layer_normalization_4/mul_1:z:0*
N*
T0*
_output_shapes
:{
6model/mpn/sequential/layer_normalization_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    щ
0model/mpn/sequential/layer_normalization_4/zerosFill@model/mpn/sequential/layer_normalization_4/zeros/packed:output:0?model/mpn/sequential/layer_normalization_4/zeros/Const:output:0*
T0*#
_output_shapes
:         s
0model/mpn/sequential/layer_normalization_4/ConstConst*
_output_shapes
: *
dtype0*
valueB u
2model/mpn/sequential/layer_normalization_4/Const_1Const*
_output_shapes
: *
dtype0*
valueB е
;model/mpn/sequential/layer_normalization_4/FusedBatchNormV3FusedBatchNormV3;model/mpn/sequential/layer_normalization_4/Reshape:output:08model/mpn/sequential/layer_normalization_4/ones:output:09model/mpn/sequential/layer_normalization_4/zeros:output:09model/mpn/sequential/layer_normalization_4/Const:output:0;model/mpn/sequential/layer_normalization_4/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:         `:         :         :         :         :*
data_formatNCHW*
epsilon%oГ:·
4model/mpn/sequential/layer_normalization_4/Reshape_1Reshape?model/mpn/sequential/layer_normalization_4/FusedBatchNormV3:y:09model/mpn/sequential/layer_normalization_4/Shape:output:0*
T0*4
_output_shapes"
 :                  `─
?model/mpn/sequential/layer_normalization_4/mul_3/ReadVariableOpReadVariableOpHmodel_mpn_sequential_layer_normalization_4_mul_3_readvariableop_resource*
_output_shapes
:`*
dtype0■
0model/mpn/sequential/layer_normalization_4/mul_3Mul=model/mpn/sequential/layer_normalization_4/Reshape_1:output:0Gmodel/mpn/sequential/layer_normalization_4/mul_3/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `└
=model/mpn/sequential/layer_normalization_4/add/ReadVariableOpReadVariableOpFmodel_mpn_sequential_layer_normalization_4_add_readvariableop_resource*
_output_shapes
:`*
dtype0є
.model/mpn/sequential/layer_normalization_4/addAddV24model/mpn/sequential/layer_normalization_4/mul_3:z:0Emodel/mpn/sequential/layer_normalization_4/add/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `о
model/mpn/mulMul2model/mpn/sequential/layer_normalization_4/add:z:0%model/tf.ones_like/ones_like:output:0*
T0*4
_output_shapes"
 :                  `Z
model/mpn/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :`С
model/mpn/zeros/packedPack model/mpn/strided_slice:output:0!model/mpn/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:Z
model/mpn/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    К
model/mpn/zerosFillmodel/mpn/zeros/packed:output:0model/mpn/zeros/Const:output:0*
T0*'
_output_shapes
:         `U
model/mpn/scan/ShapeShapemodel/mpn/mul:z:0*
T0*
_output_shapes
:l
"model/mpn/scan/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$model/mpn/scan/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$model/mpn/scan/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ь
model/mpn/scan/strided_sliceStridedSlicemodel/mpn/scan/Shape:output:0+model/mpn/scan/strided_slice/stack:output:0-model/mpn/scan/strided_slice/stack_1:output:0-model/mpn/scan/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
*model/mpn/scan/TensorArrayV2/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    `   ▀
model/mpn/scan/TensorArrayV2TensorListReserve3model/mpn/scan/TensorArrayV2/element_shape:output:0%model/mpn/scan/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥}
,model/mpn/scan/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       у
model/mpn/scan/TensorArrayV2_1TensorListReserve5model/mpn/scan/TensorArrayV2_1/element_shape:output:0%model/mpn/scan/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╧}
,model/mpn/scan/TensorArrayV2_2/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       у
model/mpn/scan/TensorArrayV2_2TensorListReserve5model/mpn/scan/TensorArrayV2_2/element_shape:output:0%model/mpn/scan/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Х
Dmodel/mpn/scan/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    `   В
6model/mpn/scan/TensorArrayUnstack/TensorListFromTensorTensorListFromTensormodel/mpn/mul:z:0Mmodel/mpn/scan/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ч
Fmodel/mpn/scan/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       №
8model/mpn/scan/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensorinput_3Omodel/mpn/scan/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╧Ч
Fmodel/mpn/scan/TensorArrayUnstack_2/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       №
8model/mpn/scan/TensorArrayUnstack_2/TensorListFromTensorTensorListFromTensorinput_4Omodel/mpn/scan/TensorArrayUnstack_2/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥}
,model/mpn/scan/TensorArrayV2_3/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    `   у
model/mpn/scan/TensorArrayV2_3TensorListReserve5model/mpn/scan/TensorArrayV2_3/element_shape:output:0%model/mpn/scan/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥V
model/mpn/scan/ConstConst*
_output_shapes
: *
dtype0*
value	B : c
!model/mpn/scan/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : с
model/mpn/scan/whileStatelessWhile*model/mpn/scan/while/loop_counter:output:0%model/mpn/scan/strided_slice:output:0model/mpn/scan/Const:output:0model/mpn/zeros:output:0'model/mpn/scan/TensorArrayV2_3:handle:0%model/mpn/scan/strided_slice:output:0Fmodel/mpn/scan/TensorArrayUnstack/TensorListFromTensor:output_handle:0Hmodel/mpn/scan/TensorArrayUnstack_1/TensorListFromTensor:output_handle:0Hmodel/mpn/scan/TensorArrayUnstack_2/TensorListFromTensor:output_handle:0 model/mpn/strided_slice:output:0*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : :         `: : : : : : * 
_read_only_resource_inputs
 *
_stateful_parallelism( *+
body#R!
model_mpn_scan_while_body_15137*+
cond#R!
model_mpn_scan_while_cond_15136*8
output_shapes'
%: : : :         `: : : : : : Р
?model/mpn/scan/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    `   °
1model/mpn/scan/TensorArrayV2Stack/TensorListStackTensorListStackmodel/mpn/scan/while:output:4Hmodel/mpn/scan/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  `*
element_dtype0i
model/mpn/lambda_1/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
         °
model/mpn/lambda_1/concatConcatV2#model/layer_normalization_2/add:z:0:model/mpn/scan/TensorArrayV2Stack/TensorListStack:tensor:0'model/mpn/lambda_1/concat/axis:output:0*
N*
T0*5
_output_shapes#
!:                  └Я
*model/mpn/dense_1/Tensordot/ReadVariableOpReadVariableOp3model_mpn_dense_1_tensordot_readvariableop_resource*
_output_shapes
:	└`*
dtype0j
 model/mpn/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:q
 model/mpn/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       s
!model/mpn/dense_1/Tensordot/ShapeShape"model/mpn/lambda_1/concat:output:0*
T0*
_output_shapes
:k
)model/mpn/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Г
$model/mpn/dense_1/Tensordot/GatherV2GatherV2*model/mpn/dense_1/Tensordot/Shape:output:0)model/mpn/dense_1/Tensordot/free:output:02model/mpn/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:m
+model/mpn/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : З
&model/mpn/dense_1/Tensordot/GatherV2_1GatherV2*model/mpn/dense_1/Tensordot/Shape:output:0)model/mpn/dense_1/Tensordot/axes:output:04model/mpn/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:k
!model/mpn/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: д
 model/mpn/dense_1/Tensordot/ProdProd-model/mpn/dense_1/Tensordot/GatherV2:output:0*model/mpn/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: m
#model/mpn/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: к
"model/mpn/dense_1/Tensordot/Prod_1Prod/model/mpn/dense_1/Tensordot/GatherV2_1:output:0,model/mpn/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: i
'model/mpn/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ф
"model/mpn/dense_1/Tensordot/concatConcatV2)model/mpn/dense_1/Tensordot/free:output:0)model/mpn/dense_1/Tensordot/axes:output:00model/mpn/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:п
!model/mpn/dense_1/Tensordot/stackPack)model/mpn/dense_1/Tensordot/Prod:output:0+model/mpn/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:├
%model/mpn/dense_1/Tensordot/transpose	Transpose"model/mpn/lambda_1/concat:output:0+model/mpn/dense_1/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:                  └└
#model/mpn/dense_1/Tensordot/ReshapeReshape)model/mpn/dense_1/Tensordot/transpose:y:0*model/mpn/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  └
"model/mpn/dense_1/Tensordot/MatMulMatMul,model/mpn/dense_1/Tensordot/Reshape:output:02model/mpn/dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `m
#model/mpn/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:`k
)model/mpn/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : я
$model/mpn/dense_1/Tensordot/concat_1ConcatV2-model/mpn/dense_1/Tensordot/GatherV2:output:0,model/mpn/dense_1/Tensordot/Const_2:output:02model/mpn/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:┬
model/mpn/dense_1/TensordotReshape,model/mpn/dense_1/Tensordot/MatMul:product:0-model/mpn/dense_1/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  `Ц
(model/mpn/dense_1/BiasAdd/ReadVariableOpReadVariableOp1model_mpn_dense_1_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0╗
model/mpn/dense_1/BiasAddBiasAdd$model/mpn/dense_1/Tensordot:output:00model/mpn/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `m
(model/mpn/sequential_1/lambda/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?├
&model/mpn/sequential_1/lambda/Gelu/mulMul1model/mpn/sequential_1/lambda/Gelu/mul/x:output:0"model/mpn/dense_1/BiasAdd:output:0*
T0*4
_output_shapes"
 :                  `n
)model/mpn/sequential_1/lambda/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *є╡?╠
*model/mpn/sequential_1/lambda/Gelu/truedivRealDiv"model/mpn/dense_1/BiasAdd:output:02model/mpn/sequential_1/lambda/Gelu/Cast/x:output:0*
T0*4
_output_shapes"
 :                  `Ь
&model/mpn/sequential_1/lambda/Gelu/ErfErf.model/mpn/sequential_1/lambda/Gelu/truediv:z:0*
T0*4
_output_shapes"
 :                  `m
(model/mpn/sequential_1/lambda/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?═
&model/mpn/sequential_1/lambda/Gelu/addAddV21model/mpn/sequential_1/lambda/Gelu/add/x:output:0*model/mpn/sequential_1/lambda/Gelu/Erf:y:0*
T0*4
_output_shapes"
 :                  `╞
(model/mpn/sequential_1/lambda/Gelu/mul_1Mul*model/mpn/sequential_1/lambda/Gelu/mul:z:0*model/mpn/sequential_1/lambda/Gelu/add:z:0*
T0*4
_output_shapes"
 :                  `О
2model/mpn/sequential_1/layer_normalization_5/ShapeShape,model/mpn/sequential_1/lambda/Gelu/mul_1:z:0*
T0*
_output_shapes
:К
@model/mpn/sequential_1/layer_normalization_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: М
Bmodel/mpn/sequential_1/layer_normalization_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:М
Bmodel/mpn/sequential_1/layer_normalization_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:▓
:model/mpn/sequential_1/layer_normalization_5/strided_sliceStridedSlice;model/mpn/sequential_1/layer_normalization_5/Shape:output:0Imodel/mpn/sequential_1/layer_normalization_5/strided_slice/stack:output:0Kmodel/mpn/sequential_1/layer_normalization_5/strided_slice/stack_1:output:0Kmodel/mpn/sequential_1/layer_normalization_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
2model/mpn/sequential_1/layer_normalization_5/mul/xConst*
_output_shapes
: *
dtype0*
value	B :┌
0model/mpn/sequential_1/layer_normalization_5/mulMul;model/mpn/sequential_1/layer_normalization_5/mul/x:output:0Cmodel/mpn/sequential_1/layer_normalization_5/strided_slice:output:0*
T0*
_output_shapes
: М
Bmodel/mpn/sequential_1/layer_normalization_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:О
Dmodel/mpn/sequential_1/layer_normalization_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:О
Dmodel/mpn/sequential_1/layer_normalization_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:║
<model/mpn/sequential_1/layer_normalization_5/strided_slice_1StridedSlice;model/mpn/sequential_1/layer_normalization_5/Shape:output:0Kmodel/mpn/sequential_1/layer_normalization_5/strided_slice_1/stack:output:0Mmodel/mpn/sequential_1/layer_normalization_5/strided_slice_1/stack_1:output:0Mmodel/mpn/sequential_1/layer_normalization_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask╫
2model/mpn/sequential_1/layer_normalization_5/mul_1Mul4model/mpn/sequential_1/layer_normalization_5/mul:z:0Emodel/mpn/sequential_1/layer_normalization_5/strided_slice_1:output:0*
T0*
_output_shapes
: М
Bmodel/mpn/sequential_1/layer_normalization_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:О
Dmodel/mpn/sequential_1/layer_normalization_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:О
Dmodel/mpn/sequential_1/layer_normalization_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:║
<model/mpn/sequential_1/layer_normalization_5/strided_slice_2StridedSlice;model/mpn/sequential_1/layer_normalization_5/Shape:output:0Kmodel/mpn/sequential_1/layer_normalization_5/strided_slice_2/stack:output:0Mmodel/mpn/sequential_1/layer_normalization_5/strided_slice_2/stack_1:output:0Mmodel/mpn/sequential_1/layer_normalization_5/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
4model/mpn/sequential_1/layer_normalization_5/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :р
2model/mpn/sequential_1/layer_normalization_5/mul_2Mul=model/mpn/sequential_1/layer_normalization_5/mul_2/x:output:0Emodel/mpn/sequential_1/layer_normalization_5/strided_slice_2:output:0*
T0*
_output_shapes
: ~
<model/mpn/sequential_1/layer_normalization_5/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :~
<model/mpn/sequential_1/layer_normalization_5/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :ю
:model/mpn/sequential_1/layer_normalization_5/Reshape/shapePackEmodel/mpn/sequential_1/layer_normalization_5/Reshape/shape/0:output:06model/mpn/sequential_1/layer_normalization_5/mul_1:z:06model/mpn/sequential_1/layer_normalization_5/mul_2:z:0Emodel/mpn/sequential_1/layer_normalization_5/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:ь
4model/mpn/sequential_1/layer_normalization_5/ReshapeReshape,model/mpn/sequential_1/lambda/Gelu/mul_1:z:0Cmodel/mpn/sequential_1/layer_normalization_5/Reshape/shape:output:0*
T0*/
_output_shapes
:         `ж
8model/mpn/sequential_1/layer_normalization_5/ones/packedPack6model/mpn/sequential_1/layer_normalization_5/mul_1:z:0*
N*
T0*
_output_shapes
:|
7model/mpn/sequential_1/layer_normalization_5/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?ь
1model/mpn/sequential_1/layer_normalization_5/onesFillAmodel/mpn/sequential_1/layer_normalization_5/ones/packed:output:0@model/mpn/sequential_1/layer_normalization_5/ones/Const:output:0*
T0*#
_output_shapes
:         з
9model/mpn/sequential_1/layer_normalization_5/zeros/packedPack6model/mpn/sequential_1/layer_normalization_5/mul_1:z:0*
N*
T0*
_output_shapes
:}
8model/mpn/sequential_1/layer_normalization_5/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    я
2model/mpn/sequential_1/layer_normalization_5/zerosFillBmodel/mpn/sequential_1/layer_normalization_5/zeros/packed:output:0Amodel/mpn/sequential_1/layer_normalization_5/zeros/Const:output:0*
T0*#
_output_shapes
:         u
2model/mpn/sequential_1/layer_normalization_5/ConstConst*
_output_shapes
: *
dtype0*
valueB w
4model/mpn/sequential_1/layer_normalization_5/Const_1Const*
_output_shapes
: *
dtype0*
valueB ▒
=model/mpn/sequential_1/layer_normalization_5/FusedBatchNormV3FusedBatchNormV3=model/mpn/sequential_1/layer_normalization_5/Reshape:output:0:model/mpn/sequential_1/layer_normalization_5/ones:output:0;model/mpn/sequential_1/layer_normalization_5/zeros:output:0;model/mpn/sequential_1/layer_normalization_5/Const:output:0=model/mpn/sequential_1/layer_normalization_5/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:         `:         :         :         :         :*
data_formatNCHW*
epsilon%oГ:А
6model/mpn/sequential_1/layer_normalization_5/Reshape_1ReshapeAmodel/mpn/sequential_1/layer_normalization_5/FusedBatchNormV3:y:0;model/mpn/sequential_1/layer_normalization_5/Shape:output:0*
T0*4
_output_shapes"
 :                  `╚
Amodel/mpn/sequential_1/layer_normalization_5/mul_3/ReadVariableOpReadVariableOpJmodel_mpn_sequential_1_layer_normalization_5_mul_3_readvariableop_resource*
_output_shapes
:`*
dtype0Д
2model/mpn/sequential_1/layer_normalization_5/mul_3Mul?model/mpn/sequential_1/layer_normalization_5/Reshape_1:output:0Imodel/mpn/sequential_1/layer_normalization_5/mul_3/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `─
?model/mpn/sequential_1/layer_normalization_5/add/ReadVariableOpReadVariableOpHmodel_mpn_sequential_1_layer_normalization_5_add_readvariableop_resource*
_output_shapes
:`*
dtype0∙
0model/mpn/sequential_1/layer_normalization_5/addAddV26model/mpn/sequential_1/layer_normalization_5/mul_3:z:0Gmodel/mpn/sequential_1/layer_normalization_5/add/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `ж
.model/node_prediction/Tensordot/ReadVariableOpReadVariableOp7model_node_prediction_tensordot_readvariableop_resource*
_output_shapes

:`*
dtype0n
$model/node_prediction/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:u
$model/node_prediction/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       Й
%model/node_prediction/Tensordot/ShapeShape4model/mpn/sequential_1/layer_normalization_5/add:z:0*
T0*
_output_shapes
:o
-model/node_prediction/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : У
(model/node_prediction/Tensordot/GatherV2GatherV2.model/node_prediction/Tensordot/Shape:output:0-model/node_prediction/Tensordot/free:output:06model/node_prediction/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:q
/model/node_prediction/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ч
*model/node_prediction/Tensordot/GatherV2_1GatherV2.model/node_prediction/Tensordot/Shape:output:0-model/node_prediction/Tensordot/axes:output:08model/node_prediction/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:o
%model/node_prediction/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ░
$model/node_prediction/Tensordot/ProdProd1model/node_prediction/Tensordot/GatherV2:output:0.model/node_prediction/Tensordot/Const:output:0*
T0*
_output_shapes
: q
'model/node_prediction/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ╢
&model/node_prediction/Tensordot/Prod_1Prod3model/node_prediction/Tensordot/GatherV2_1:output:00model/node_prediction/Tensordot/Const_1:output:0*
T0*
_output_shapes
: m
+model/node_prediction/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ї
&model/node_prediction/Tensordot/concatConcatV2-model/node_prediction/Tensordot/free:output:0-model/node_prediction/Tensordot/axes:output:04model/node_prediction/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:╗
%model/node_prediction/Tensordot/stackPack-model/node_prediction/Tensordot/Prod:output:0/model/node_prediction/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:▄
)model/node_prediction/Tensordot/transpose	Transpose4model/mpn/sequential_1/layer_normalization_5/add:z:0/model/node_prediction/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :                  `╠
'model/node_prediction/Tensordot/ReshapeReshape-model/node_prediction/Tensordot/transpose:y:0.model/node_prediction/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  ╠
&model/node_prediction/Tensordot/MatMulMatMul0model/node_prediction/Tensordot/Reshape:output:06model/node_prediction/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         q
'model/node_prediction/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:o
-model/node_prediction/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :  
(model/node_prediction/Tensordot/concat_1ConcatV21model/node_prediction/Tensordot/GatherV2:output:00model/node_prediction/Tensordot/Const_2:output:06model/node_prediction/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:╬
model/node_prediction/TensordotReshape0model/node_prediction/Tensordot/MatMul:product:01model/node_prediction/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  Ю
,model/node_prediction/BiasAdd/ReadVariableOpReadVariableOp5model_node_prediction_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0╟
model/node_prediction/BiasAddBiasAdd(model/node_prediction/Tensordot:output:04model/node_prediction/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  В
IdentityIdentity&model/node_prediction/BiasAdd:output:0^NoOp*
T0*4
_output_shapes"
 :                  ╠

NoOpNoOp'^model/edge_ide1/BiasAdd/ReadVariableOp)^model/edge_ide1/Tensordot/ReadVariableOp'^model/edge_ide2/BiasAdd/ReadVariableOp)^model/edge_ide2/Tensordot/ReadVariableOp-^model/layer_normalization/add/ReadVariableOp/^model/layer_normalization/mul_3/ReadVariableOp/^model/layer_normalization_1/add/ReadVariableOp1^model/layer_normalization_1/mul_3/ReadVariableOp/^model/layer_normalization_2/add/ReadVariableOp1^model/layer_normalization_2/mul_3/ReadVariableOp/^model/layer_normalization_3/add/ReadVariableOp1^model/layer_normalization_3/mul_3/ReadVariableOp)^model/mpn/dense_1/BiasAdd/ReadVariableOp+^model/mpn/dense_1/Tensordot/ReadVariableOp2^model/mpn/sequential/dense/BiasAdd/ReadVariableOp4^model/mpn/sequential/dense/Tensordot/ReadVariableOp>^model/mpn/sequential/layer_normalization_4/add/ReadVariableOp@^model/mpn/sequential/layer_normalization_4/mul_3/ReadVariableOp@^model/mpn/sequential_1/layer_normalization_5/add/ReadVariableOpB^model/mpn/sequential_1/layer_normalization_5/mul_3/ReadVariableOp'^model/node_ide1/BiasAdd/ReadVariableOp)^model/node_ide1/Tensordot/ReadVariableOp'^model/node_ide2/BiasAdd/ReadVariableOp)^model/node_ide2/Tensordot/ReadVariableOp-^model/node_prediction/BiasAdd/ReadVariableOp/^model/node_prediction/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*╔
_input_shapes╖
┤:                  :                  :                  :                  : : : : : : : : : : : : : : : : : : : : : : : : : : 2P
&model/edge_ide1/BiasAdd/ReadVariableOp&model/edge_ide1/BiasAdd/ReadVariableOp2T
(model/edge_ide1/Tensordot/ReadVariableOp(model/edge_ide1/Tensordot/ReadVariableOp2P
&model/edge_ide2/BiasAdd/ReadVariableOp&model/edge_ide2/BiasAdd/ReadVariableOp2T
(model/edge_ide2/Tensordot/ReadVariableOp(model/edge_ide2/Tensordot/ReadVariableOp2\
,model/layer_normalization/add/ReadVariableOp,model/layer_normalization/add/ReadVariableOp2`
.model/layer_normalization/mul_3/ReadVariableOp.model/layer_normalization/mul_3/ReadVariableOp2`
.model/layer_normalization_1/add/ReadVariableOp.model/layer_normalization_1/add/ReadVariableOp2d
0model/layer_normalization_1/mul_3/ReadVariableOp0model/layer_normalization_1/mul_3/ReadVariableOp2`
.model/layer_normalization_2/add/ReadVariableOp.model/layer_normalization_2/add/ReadVariableOp2d
0model/layer_normalization_2/mul_3/ReadVariableOp0model/layer_normalization_2/mul_3/ReadVariableOp2`
.model/layer_normalization_3/add/ReadVariableOp.model/layer_normalization_3/add/ReadVariableOp2d
0model/layer_normalization_3/mul_3/ReadVariableOp0model/layer_normalization_3/mul_3/ReadVariableOp2T
(model/mpn/dense_1/BiasAdd/ReadVariableOp(model/mpn/dense_1/BiasAdd/ReadVariableOp2X
*model/mpn/dense_1/Tensordot/ReadVariableOp*model/mpn/dense_1/Tensordot/ReadVariableOp2f
1model/mpn/sequential/dense/BiasAdd/ReadVariableOp1model/mpn/sequential/dense/BiasAdd/ReadVariableOp2j
3model/mpn/sequential/dense/Tensordot/ReadVariableOp3model/mpn/sequential/dense/Tensordot/ReadVariableOp2~
=model/mpn/sequential/layer_normalization_4/add/ReadVariableOp=model/mpn/sequential/layer_normalization_4/add/ReadVariableOp2В
?model/mpn/sequential/layer_normalization_4/mul_3/ReadVariableOp?model/mpn/sequential/layer_normalization_4/mul_3/ReadVariableOp2В
?model/mpn/sequential_1/layer_normalization_5/add/ReadVariableOp?model/mpn/sequential_1/layer_normalization_5/add/ReadVariableOp2Ж
Amodel/mpn/sequential_1/layer_normalization_5/mul_3/ReadVariableOpAmodel/mpn/sequential_1/layer_normalization_5/mul_3/ReadVariableOp2P
&model/node_ide1/BiasAdd/ReadVariableOp&model/node_ide1/BiasAdd/ReadVariableOp2T
(model/node_ide1/Tensordot/ReadVariableOp(model/node_ide1/Tensordot/ReadVariableOp2P
&model/node_ide2/BiasAdd/ReadVariableOp&model/node_ide2/BiasAdd/ReadVariableOp2T
(model/node_ide2/Tensordot/ReadVariableOp(model/node_ide2/Tensordot/ReadVariableOp2\
,model/node_prediction/BiasAdd/ReadVariableOp,model/node_prediction/BiasAdd/ReadVariableOp2`
.model/node_prediction/Tensordot/ReadVariableOp.model/node_prediction/Tensordot/ReadVariableOp:] Y
4
_output_shapes"
 :                  
!
_user_specified_name	input_1:]Y
4
_output_shapes"
 :                  
!
_user_specified_name	input_2:]Y
4
_output_shapes"
 :                  
!
_user_specified_name	input_3:]Y
4
_output_shapes"
 :                  
!
_user_specified_name	input_4
Ы
ў
scan_while_cond_19694&
"scan_while_scan_while_loop_counter!
scan_while_scan_strided_slice
scan_while_placeholder
scan_while_placeholder_1
scan_while_placeholder_2&
"scan_while_less_scan_strided_slice=
9scan_while_scan_while_cond_19694___redundant_placeholder0=
9scan_while_scan_while_cond_19694___redundant_placeholder1=
9scan_while_scan_while_cond_19694___redundant_placeholder2=
9scan_while_scan_while_cond_19694___redundant_placeholder3
scan_while_identity
t
scan/while/LessLessscan_while_placeholder"scan_while_less_scan_strided_slice*
T0*
_output_shapes
: }
scan/while/Less_1Less"scan_while_scan_while_loop_counterscan_while_scan_strided_slice*
T0*
_output_shapes
: g
scan/while/LogicalAnd
LogicalAndscan/while/Less_1:z:0scan/while/Less:z:0*
_output_shapes
: [
scan/while/IdentityIdentityscan/while/LogicalAnd:z:0*
T0
*
_output_shapes
: "3
scan_while_identityscan/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : :         `: : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         `:

_output_shapes
: :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
:
нА
Г	
>__inference_mpn_layer_call_and_return_conditional_losses_16786

inputs
inputs_1
inputs_2
inputs_3
inputs_4E
2sequential_dense_tensordot_readvariableop_resource:	а`>
0sequential_dense_biasadd_readvariableop_resource:`L
>sequential_layer_normalization_4_mul_3_readvariableop_resource:`J
<sequential_layer_normalization_4_add_readvariableop_resource:`<
)dense_1_tensordot_readvariableop_resource:	└`5
'dense_1_biasadd_readvariableop_resource:`N
@sequential_1_layer_normalization_5_mul_3_readvariableop_resource:`L
>sequential_1_layer_normalization_5_add_readvariableop_resource:`
identity

identity_1

identity_2

identity_3

identity_4Ивdense_1/BiasAdd/ReadVariableOpв dense_1/Tensordot/ReadVariableOpв'sequential/dense/BiasAdd/ReadVariableOpв)sequential/dense/Tensordot/ReadVariableOpв3sequential/layer_normalization_4/add/ReadVariableOpв5sequential/layer_normalization_4/mul_3/ReadVariableOpв5sequential_1/layer_normalization_5/add/ReadVariableOpв7sequential_1/layer_normalization_5/mul_3/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Shape_1Shapeinputs_2*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask=
Shape_2Shapeinputs*
T0*
_output_shapes
:_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_2StridedSliceShape_2:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskO
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :╢
GatherV2GatherV2inputsinputs_2GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*8
_output_shapes&
$:"                  `*

batch_dimsR
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :└С
Reshape/shapePackstrided_slice_2:output:0strided_slice_1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:}
ReshapeReshapeGatherV2:output:0Reshape/shape:output:0*
T0*5
_output_shapes#
!:                  └V
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
         Н
concatConcatV2Reshape:output:0inputs_1concat/axis:output:0*
N*
T0*5
_output_shapes#
!:                  аЭ
)sequential/dense/Tensordot/ReadVariableOpReadVariableOp2sequential_dense_tensordot_readvariableop_resource*
_output_shapes
:	а`*
dtype0i
sequential/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:p
sequential/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       _
 sequential/dense/Tensordot/ShapeShapeconcat:output:0*
T0*
_output_shapes
:j
(sequential/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :  
#sequential/dense/Tensordot/GatherV2GatherV2)sequential/dense/Tensordot/Shape:output:0(sequential/dense/Tensordot/free:output:01sequential/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
*sequential/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Г
%sequential/dense/Tensordot/GatherV2_1GatherV2)sequential/dense/Tensordot/Shape:output:0(sequential/dense/Tensordot/axes:output:03sequential/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:j
 sequential/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: б
sequential/dense/Tensordot/ProdProd,sequential/dense/Tensordot/GatherV2:output:0)sequential/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: l
"sequential/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: з
!sequential/dense/Tensordot/Prod_1Prod.sequential/dense/Tensordot/GatherV2_1:output:0+sequential/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: h
&sequential/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : р
!sequential/dense/Tensordot/concatConcatV2(sequential/dense/Tensordot/free:output:0(sequential/dense/Tensordot/axes:output:0/sequential/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:м
 sequential/dense/Tensordot/stackPack(sequential/dense/Tensordot/Prod:output:0*sequential/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:о
$sequential/dense/Tensordot/transpose	Transposeconcat:output:0*sequential/dense/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:                  а╜
"sequential/dense/Tensordot/ReshapeReshape(sequential/dense/Tensordot/transpose:y:0)sequential/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  ╜
!sequential/dense/Tensordot/MatMulMatMul+sequential/dense/Tensordot/Reshape:output:01sequential/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `l
"sequential/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:`j
(sequential/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ы
#sequential/dense/Tensordot/concat_1ConcatV2,sequential/dense/Tensordot/GatherV2:output:0+sequential/dense/Tensordot/Const_2:output:01sequential/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:┐
sequential/dense/TensordotReshape+sequential/dense/Tensordot/MatMul:product:0,sequential/dense/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  `Ф
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0╕
sequential/dense/BiasAddBiasAdd#sequential/dense/Tensordot:output:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `ц
!sequential/lambda/PartitionedCallPartitionedCall!sequential/dense/BiasAdd:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  `* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_lambda_layer_call_and_return_conditional_losses_15461А
&sequential/layer_normalization_4/ShapeShape*sequential/lambda/PartitionedCall:output:0*
T0*
_output_shapes
:~
4sequential/layer_normalization_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: А
6sequential/layer_normalization_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:А
6sequential/layer_normalization_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ў
.sequential/layer_normalization_4/strided_sliceStridedSlice/sequential/layer_normalization_4/Shape:output:0=sequential/layer_normalization_4/strided_slice/stack:output:0?sequential/layer_normalization_4/strided_slice/stack_1:output:0?sequential/layer_normalization_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
&sequential/layer_normalization_4/mul/xConst*
_output_shapes
: *
dtype0*
value	B :╢
$sequential/layer_normalization_4/mulMul/sequential/layer_normalization_4/mul/x:output:07sequential/layer_normalization_4/strided_slice:output:0*
T0*
_output_shapes
: А
6sequential/layer_normalization_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:В
8sequential/layer_normalization_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:В
8sequential/layer_normalization_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:■
0sequential/layer_normalization_4/strided_slice_1StridedSlice/sequential/layer_normalization_4/Shape:output:0?sequential/layer_normalization_4/strided_slice_1/stack:output:0Asequential/layer_normalization_4/strided_slice_1/stack_1:output:0Asequential/layer_normalization_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask│
&sequential/layer_normalization_4/mul_1Mul(sequential/layer_normalization_4/mul:z:09sequential/layer_normalization_4/strided_slice_1:output:0*
T0*
_output_shapes
: А
6sequential/layer_normalization_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:В
8sequential/layer_normalization_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:В
8sequential/layer_normalization_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:■
0sequential/layer_normalization_4/strided_slice_2StridedSlice/sequential/layer_normalization_4/Shape:output:0?sequential/layer_normalization_4/strided_slice_2/stack:output:0Asequential/layer_normalization_4/strided_slice_2/stack_1:output:0Asequential/layer_normalization_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(sequential/layer_normalization_4/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :╝
&sequential/layer_normalization_4/mul_2Mul1sequential/layer_normalization_4/mul_2/x:output:09sequential/layer_normalization_4/strided_slice_2:output:0*
T0*
_output_shapes
: r
0sequential/layer_normalization_4/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :r
0sequential/layer_normalization_4/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :▓
.sequential/layer_normalization_4/Reshape/shapePack9sequential/layer_normalization_4/Reshape/shape/0:output:0*sequential/layer_normalization_4/mul_1:z:0*sequential/layer_normalization_4/mul_2:z:09sequential/layer_normalization_4/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:╥
(sequential/layer_normalization_4/ReshapeReshape*sequential/lambda/PartitionedCall:output:07sequential/layer_normalization_4/Reshape/shape:output:0*
T0*/
_output_shapes
:         `О
,sequential/layer_normalization_4/ones/packedPack*sequential/layer_normalization_4/mul_1:z:0*
N*
T0*
_output_shapes
:p
+sequential/layer_normalization_4/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╚
%sequential/layer_normalization_4/onesFill5sequential/layer_normalization_4/ones/packed:output:04sequential/layer_normalization_4/ones/Const:output:0*
T0*#
_output_shapes
:         П
-sequential/layer_normalization_4/zeros/packedPack*sequential/layer_normalization_4/mul_1:z:0*
N*
T0*
_output_shapes
:q
,sequential/layer_normalization_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ╦
&sequential/layer_normalization_4/zerosFill6sequential/layer_normalization_4/zeros/packed:output:05sequential/layer_normalization_4/zeros/Const:output:0*
T0*#
_output_shapes
:         i
&sequential/layer_normalization_4/ConstConst*
_output_shapes
: *
dtype0*
valueB k
(sequential/layer_normalization_4/Const_1Const*
_output_shapes
: *
dtype0*
valueB щ
1sequential/layer_normalization_4/FusedBatchNormV3FusedBatchNormV31sequential/layer_normalization_4/Reshape:output:0.sequential/layer_normalization_4/ones:output:0/sequential/layer_normalization_4/zeros:output:0/sequential/layer_normalization_4/Const:output:01sequential/layer_normalization_4/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:         `:         :         :         :         :*
data_formatNCHW*
epsilon%oГ:▄
*sequential/layer_normalization_4/Reshape_1Reshape5sequential/layer_normalization_4/FusedBatchNormV3:y:0/sequential/layer_normalization_4/Shape:output:0*
T0*4
_output_shapes"
 :                  `░
5sequential/layer_normalization_4/mul_3/ReadVariableOpReadVariableOp>sequential_layer_normalization_4_mul_3_readvariableop_resource*
_output_shapes
:`*
dtype0р
&sequential/layer_normalization_4/mul_3Mul3sequential/layer_normalization_4/Reshape_1:output:0=sequential/layer_normalization_4/mul_3/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `м
3sequential/layer_normalization_4/add/ReadVariableOpReadVariableOp<sequential_layer_normalization_4_add_readvariableop_resource*
_output_shapes
:`*
dtype0╒
$sequential/layer_normalization_4/addAddV2*sequential/layer_normalization_4/mul_3:z:0;sequential/layer_normalization_4/add/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `}
mulMul(sequential/layer_normalization_4/add:z:0inputs_3*
T0*4
_output_shapes"
 :                  `P
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :`s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         `A

scan/ShapeShapemul:z:0*
T0*
_output_shapes
:b
scan/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: d
scan/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:d
scan/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ъ
scan/strided_sliceStridedSlicescan/Shape:output:0!scan/strided_slice/stack:output:0#scan/strided_slice/stack_1:output:0#scan/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskq
 scan/TensorArrayV2/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    `   ┴
scan/TensorArrayV2TensorListReserve)scan/TensorArrayV2/element_shape:output:0scan/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥s
"scan/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ┼
scan/TensorArrayV2_1TensorListReserve+scan/TensorArrayV2_1/element_shape:output:0scan/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╧s
"scan/TensorArrayV2_2/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ┼
scan/TensorArrayV2_2TensorListReserve+scan/TensorArrayV2_2/element_shape:output:0scan/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Л
:scan/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    `   ф
,scan/TensorArrayUnstack/TensorListFromTensorTensorListFromTensormul:z:0Cscan/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Н
<scan/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       щ
.scan/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensorinputs_2Escan/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╧Н
<scan/TensorArrayUnstack_2/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       щ
.scan/TensorArrayUnstack_2/TensorListFromTensorTensorListFromTensorinputs_4Escan/TensorArrayUnstack_2/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥s
"scan/TensorArrayV2_3/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    `   ┼
scan/TensorArrayV2_3TensorListReserve+scan/TensorArrayV2_3/element_shape:output:0scan/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥L

scan/ConstConst*
_output_shapes
: *
dtype0*
value	B : Y
scan/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ▀

scan/whileStatelessWhile scan/while/loop_counter:output:0scan/strided_slice:output:0scan/Const:output:0zeros:output:0scan/TensorArrayV2_3:handle:0scan/strided_slice:output:0<scan/TensorArrayUnstack/TensorListFromTensor:output_handle:0>scan/TensorArrayUnstack_1/TensorListFromTensor:output_handle:0>scan/TensorArrayUnstack_2/TensorListFromTensor:output_handle:0strided_slice:output:0*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : :         `: : : : : : * 
_read_only_resource_inputs
 *
_stateful_parallelism( *!
bodyR
scan_while_body_16643*!
condR
scan_while_cond_16642*8
output_shapes'
%: : : :         `: : : : : : Ж
5scan/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    `   ┌
'scan/TensorArrayV2Stack/TensorListStackTensorListStackscan/while:output:4>scan/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  `*
element_dtype0_
lambda_1/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
         ╜
lambda_1/concatConcatV2inputs0scan/TensorArrayV2Stack/TensorListStack:tensor:0lambda_1/concat/axis:output:0*
N*
T0*5
_output_shapes#
!:                  └Л
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource*
_output_shapes
:	└`*
dtype0`
dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       _
dense_1/Tensordot/ShapeShapelambda_1/concat:output:0*
T0*
_output_shapes
:a
dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : █
dense_1/Tensordot/GatherV2GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/free:output:0(dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ▀
dense_1/Tensordot/GatherV2_1GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/axes:output:0*dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: Ж
dense_1/Tensordot/ProdProd#dense_1/Tensordot/GatherV2:output:0 dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: М
dense_1/Tensordot/Prod_1Prod%dense_1/Tensordot/GatherV2_1:output:0"dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ╝
dense_1/Tensordot/concatConcatV2dense_1/Tensordot/free:output:0dense_1/Tensordot/axes:output:0&dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:С
dense_1/Tensordot/stackPackdense_1/Tensordot/Prod:output:0!dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:е
dense_1/Tensordot/transpose	Transposelambda_1/concat:output:0!dense_1/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:                  └в
dense_1/Tensordot/ReshapeReshapedense_1/Tensordot/transpose:y:0 dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  в
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0(dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `c
dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:`a
dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ╟
dense_1/Tensordot/concat_1ConcatV2#dense_1/Tensordot/GatherV2:output:0"dense_1/Tensordot/Const_2:output:0(dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:д
dense_1/TensordotReshape"dense_1/Tensordot/MatMul:product:0#dense_1/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  `В
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0Э
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `▀
#sequential_1/lambda/PartitionedCallPartitionedCalldense_1/BiasAdd:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  `* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_lambda_layer_call_and_return_conditional_losses_15461Д
(sequential_1/layer_normalization_5/ShapeShape,sequential_1/lambda/PartitionedCall:output:0*
T0*
_output_shapes
:А
6sequential_1/layer_normalization_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: В
8sequential_1/layer_normalization_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:В
8sequential_1/layer_normalization_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:А
0sequential_1/layer_normalization_5/strided_sliceStridedSlice1sequential_1/layer_normalization_5/Shape:output:0?sequential_1/layer_normalization_5/strided_slice/stack:output:0Asequential_1/layer_normalization_5/strided_slice/stack_1:output:0Asequential_1/layer_normalization_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(sequential_1/layer_normalization_5/mul/xConst*
_output_shapes
: *
dtype0*
value	B :╝
&sequential_1/layer_normalization_5/mulMul1sequential_1/layer_normalization_5/mul/x:output:09sequential_1/layer_normalization_5/strided_slice:output:0*
T0*
_output_shapes
: В
8sequential_1/layer_normalization_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:Д
:sequential_1/layer_normalization_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Д
:sequential_1/layer_normalization_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
2sequential_1/layer_normalization_5/strided_slice_1StridedSlice1sequential_1/layer_normalization_5/Shape:output:0Asequential_1/layer_normalization_5/strided_slice_1/stack:output:0Csequential_1/layer_normalization_5/strided_slice_1/stack_1:output:0Csequential_1/layer_normalization_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask╣
(sequential_1/layer_normalization_5/mul_1Mul*sequential_1/layer_normalization_5/mul:z:0;sequential_1/layer_normalization_5/strided_slice_1:output:0*
T0*
_output_shapes
: В
8sequential_1/layer_normalization_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:Д
:sequential_1/layer_normalization_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Д
:sequential_1/layer_normalization_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
2sequential_1/layer_normalization_5/strided_slice_2StridedSlice1sequential_1/layer_normalization_5/Shape:output:0Asequential_1/layer_normalization_5/strided_slice_2/stack:output:0Csequential_1/layer_normalization_5/strided_slice_2/stack_1:output:0Csequential_1/layer_normalization_5/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
*sequential_1/layer_normalization_5/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :┬
(sequential_1/layer_normalization_5/mul_2Mul3sequential_1/layer_normalization_5/mul_2/x:output:0;sequential_1/layer_normalization_5/strided_slice_2:output:0*
T0*
_output_shapes
: t
2sequential_1/layer_normalization_5/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :t
2sequential_1/layer_normalization_5/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :╝
0sequential_1/layer_normalization_5/Reshape/shapePack;sequential_1/layer_normalization_5/Reshape/shape/0:output:0,sequential_1/layer_normalization_5/mul_1:z:0,sequential_1/layer_normalization_5/mul_2:z:0;sequential_1/layer_normalization_5/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:╪
*sequential_1/layer_normalization_5/ReshapeReshape,sequential_1/lambda/PartitionedCall:output:09sequential_1/layer_normalization_5/Reshape/shape:output:0*
T0*/
_output_shapes
:         `Т
.sequential_1/layer_normalization_5/ones/packedPack,sequential_1/layer_normalization_5/mul_1:z:0*
N*
T0*
_output_shapes
:r
-sequential_1/layer_normalization_5/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╬
'sequential_1/layer_normalization_5/onesFill7sequential_1/layer_normalization_5/ones/packed:output:06sequential_1/layer_normalization_5/ones/Const:output:0*
T0*#
_output_shapes
:         У
/sequential_1/layer_normalization_5/zeros/packedPack,sequential_1/layer_normalization_5/mul_1:z:0*
N*
T0*
_output_shapes
:s
.sequential_1/layer_normalization_5/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ╤
(sequential_1/layer_normalization_5/zerosFill8sequential_1/layer_normalization_5/zeros/packed:output:07sequential_1/layer_normalization_5/zeros/Const:output:0*
T0*#
_output_shapes
:         k
(sequential_1/layer_normalization_5/ConstConst*
_output_shapes
: *
dtype0*
valueB m
*sequential_1/layer_normalization_5/Const_1Const*
_output_shapes
: *
dtype0*
valueB ї
3sequential_1/layer_normalization_5/FusedBatchNormV3FusedBatchNormV33sequential_1/layer_normalization_5/Reshape:output:00sequential_1/layer_normalization_5/ones:output:01sequential_1/layer_normalization_5/zeros:output:01sequential_1/layer_normalization_5/Const:output:03sequential_1/layer_normalization_5/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:         `:         :         :         :         :*
data_formatNCHW*
epsilon%oГ:т
,sequential_1/layer_normalization_5/Reshape_1Reshape7sequential_1/layer_normalization_5/FusedBatchNormV3:y:01sequential_1/layer_normalization_5/Shape:output:0*
T0*4
_output_shapes"
 :                  `┤
7sequential_1/layer_normalization_5/mul_3/ReadVariableOpReadVariableOp@sequential_1_layer_normalization_5_mul_3_readvariableop_resource*
_output_shapes
:`*
dtype0ц
(sequential_1/layer_normalization_5/mul_3Mul5sequential_1/layer_normalization_5/Reshape_1:output:0?sequential_1/layer_normalization_5/mul_3/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `░
5sequential_1/layer_normalization_5/add/ReadVariableOpReadVariableOp>sequential_1_layer_normalization_5_add_readvariableop_resource*
_output_shapes
:`*
dtype0█
&sequential_1/layer_normalization_5/addAddV2,sequential_1/layer_normalization_5/mul_3:z:0=sequential_1/layer_normalization_5/add/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `Ж
IdentityIdentity*sequential_1/layer_normalization_5/add:z:0^NoOp*
T0*4
_output_shapes"
 :                  `e

Identity_1Identitymul:z:0^NoOp*
T0*4
_output_shapes"
 :                  `f

Identity_2Identityinputs_2^NoOp*
T0*4
_output_shapes"
 :                  f

Identity_3Identityinputs_3^NoOp*
T0*4
_output_shapes"
 :                  f

Identity_4Identityinputs_4^NoOp*
T0*4
_output_shapes"
 :                  └
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp*^sequential/dense/Tensordot/ReadVariableOp4^sequential/layer_normalization_4/add/ReadVariableOp6^sequential/layer_normalization_4/mul_3/ReadVariableOp6^sequential_1/layer_normalization_5/add/ReadVariableOp8^sequential_1/layer_normalization_5/mul_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*┼
_input_shapes│
░:                  `:                  `:                  :                  :                  : : : : : : : : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_1/Tensordot/ReadVariableOp dense_1/Tensordot/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2V
)sequential/dense/Tensordot/ReadVariableOp)sequential/dense/Tensordot/ReadVariableOp2j
3sequential/layer_normalization_4/add/ReadVariableOp3sequential/layer_normalization_4/add/ReadVariableOp2n
5sequential/layer_normalization_4/mul_3/ReadVariableOp5sequential/layer_normalization_4/mul_3/ReadVariableOp2n
5sequential_1/layer_normalization_5/add/ReadVariableOp5sequential_1/layer_normalization_5/add/ReadVariableOp2r
7sequential_1/layer_normalization_5/mul_3/ReadVariableOp7sequential_1/layer_normalization_5/mul_3/ReadVariableOp:\ X
4
_output_shapes"
 :                  `
 
_user_specified_nameinputs:\X
4
_output_shapes"
 :                  `
 
_user_specified_nameinputs:\X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs:\X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs:\X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
В
Ь
/__inference_node_prediction_layer_call_fn_19854

inputs
unknown:`
	unknown_0:
identityИвStatefulPartitionedCallь
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_node_prediction_layer_call_and_return_conditional_losses_16406|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:                  `: : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :                  `
 
_user_specified_nameinputs
Ы
ў
scan_while_cond_16210&
"scan_while_scan_while_loop_counter!
scan_while_scan_strided_slice
scan_while_placeholder
scan_while_placeholder_1
scan_while_placeholder_2&
"scan_while_less_scan_strided_slice=
9scan_while_scan_while_cond_16210___redundant_placeholder0=
9scan_while_scan_while_cond_16210___redundant_placeholder1=
9scan_while_scan_while_cond_16210___redundant_placeholder2=
9scan_while_scan_while_cond_16210___redundant_placeholder3
scan_while_identity
t
scan/while/LessLessscan_while_placeholder"scan_while_less_scan_strided_slice*
T0*
_output_shapes
: }
scan/while/Less_1Less"scan_while_scan_while_loop_counterscan_while_scan_strided_slice*
T0*
_output_shapes
: g
scan/while/LogicalAnd
LogicalAndscan/while/Less_1:z:0scan/while/Less:z:0*
_output_shapes
: [
scan/while/IdentityIdentityscan/while/LogicalAnd:z:0*
T0
*
_output_shapes
: "3
scan_while_identityscan/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : :         `: : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         `:

_output_shapes
: :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
:
М
╣
#__inference_signature_wrapper_17411
input_1
input_2
input_3
input_4
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@`
	unknown_6:`
	unknown_7:@
	unknown_8:@
	unknown_9:@`

unknown_10:`

unknown_11:`

unknown_12:`

unknown_13:`

unknown_14:`

unknown_15:	а`

unknown_16:`

unknown_17:`

unknown_18:`

unknown_19:	└`

unknown_20:`

unknown_21:`

unknown_22:`

unknown_23:`

unknown_24:
identityИвStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2input_3input_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  *<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *)
f$R"
 __inference__wrapped_model_15309|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*╔
_input_shapes╖
┤:                  :                  :                  :                  : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
4
_output_shapes"
 :                  
!
_user_specified_name	input_1:]Y
4
_output_shapes"
 :                  
!
_user_specified_name	input_2:]Y
4
_output_shapes"
 :                  
!
_user_specified_name	input_3:]Y
4
_output_shapes"
 :                  
!
_user_specified_name	input_4
╦
B
&__inference_lambda_layer_call_fn_18816

inputs
identity╣
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  `* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_lambda_layer_call_and_return_conditional_losses_15461m
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :                  `"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :                  `:\ X
4
_output_shapes"
 :                  `
 
_user_specified_nameinputs
Г
√
D__inference_edge_ide1_layer_call_and_return_conditional_losses_19008

inputs3
!tensordot_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвTensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ┐
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ь
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:В
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*4
_output_shapes"
 :                  К
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:                  К
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : з
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:М
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Е
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  @l
IdentityIdentityBiasAdd:output:0^NoOp*
T0*4
_output_shapes"
 :                  @z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:                  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
ъ5
╦
scan_while_body_19413&
"scan_while_scan_while_loop_counter!
scan_while_scan_strided_slice
scan_while_placeholder
scan_while_placeholder_1
scan_while_placeholder_2%
!scan_while_scan_strided_slice_1_0a
]scan_while_tensorarrayv2read_tensorlistgetitem_scan_tensorarrayunstack_tensorlistfromtensor_0e
ascan_while_tensorarrayv2read_1_tensorlistgetitem_scan_tensorarrayunstack_1_tensorlistfromtensor_0e
ascan_while_tensorarrayv2read_2_tensorlistgetitem_scan_tensorarrayunstack_2_tensorlistfromtensor_01
-scan_while_unsortedsegmentsum_strided_slice_0
scan_while_identity
scan_while_identity_1
scan_while_identity_2
scan_while_identity_3
scan_while_identity_4#
scan_while_scan_strided_slice_1_
[scan_while_tensorarrayv2read_tensorlistgetitem_scan_tensorarrayunstack_tensorlistfromtensorc
_scan_while_tensorarrayv2read_1_tensorlistgetitem_scan_tensorarrayunstack_1_tensorlistfromtensorc
_scan_while_tensorarrayv2read_2_tensorlistgetitem_scan_tensorarrayunstack_2_tensorlistfromtensor/
+scan_while_unsortedsegmentsum_strided_sliceН
<scan/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    `   ┐
.scan/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem]scan_while_tensorarrayv2read_tensorlistgetitem_scan_tensorarrayunstack_tensorlistfromtensor_0scan_while_placeholderEscan/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         `*
element_dtype0П
>scan/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╟
0scan/while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItemascan_while_tensorarrayv2read_1_tensorlistgetitem_scan_tensorarrayunstack_1_tensorlistfromtensor_0scan_while_placeholderGscan/while/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0П
>scan/while/TensorArrayV2Read_2/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╟
0scan/while/TensorArrayV2Read_2/TensorListGetItemTensorListGetItemascan_while_tensorarrayv2read_2_tensorlistgetitem_scan_tensorarrayunstack_2_tensorlistfromtensor_0scan_while_placeholderGscan/while/TensorArrayV2Read_2/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0o
scan/while/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       q
 scan/while/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       q
 scan/while/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ┴
scan/while/strided_sliceStridedSlice7scan/while/TensorArrayV2Read_2/TensorListGetItem:item:0'scan/while/strided_slice/stack:output:0)scan/while/strided_slice/stack_1:output:0)scan/while/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_maskб
scan/while/mulMul5scan/while/TensorArrayV2Read/TensorListGetItem:item:0!scan/while/strided_slice:output:0*
T0*'
_output_shapes
:         `q
 scan/while/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       s
"scan/while/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       s
"scan/while/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ▌
scan/while/strided_slice_1StridedSlice7scan/while/TensorArrayV2Read_1/TensorListGetItem:item:0)scan/while/strided_slice_1/stack:output:0+scan/while/strided_slice_1/stack_1:output:0+scan/while/strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask▌
scan/while/UnsortedSegmentSumUnsortedSegmentSumscan/while/mul:z:0#scan/while/strided_slice_1:output:0-scan_while_unsortedsegmentsum_strided_slice_0*
T0*
Tindices0*'
_output_shapes
:         `▐
/scan/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemscan_while_placeholder_2scan_while_placeholder&scan/while/UnsortedSegmentSum:output:0*
_output_shapes
: *
element_dtype0:щш╥R
scan/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :k
scan/while/addAddV2scan_while_placeholderscan/while/add/y:output:0*
T0*
_output_shapes
: T
scan/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :{
scan/while/add_1AddV2"scan_while_scan_while_loop_counterscan/while/add_1/y:output:0*
T0*
_output_shapes
: V
scan/while/IdentityIdentityscan/while/add_1:z:0*
T0*
_output_shapes
: a
scan/while/Identity_1Identityscan_while_scan_strided_slice*
T0*
_output_shapes
: V
scan/while/Identity_2Identityscan/while/add:z:0*
T0*
_output_shapes
: {
scan/while/Identity_3Identity&scan/while/UnsortedSegmentSum:output:0*
T0*'
_output_shapes
:         `Г
scan/while/Identity_4Identity?scan/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: "3
scan_while_identityscan/while/Identity:output:0"7
scan_while_identity_1scan/while/Identity_1:output:0"7
scan_while_identity_2scan/while/Identity_2:output:0"7
scan_while_identity_3scan/while/Identity_3:output:0"7
scan_while_identity_4scan/while/Identity_4:output:0"D
scan_while_scan_strided_slice_1!scan_while_scan_strided_slice_1_0"─
_scan_while_tensorarrayv2read_1_tensorlistgetitem_scan_tensorarrayunstack_1_tensorlistfromtensorascan_while_tensorarrayv2read_1_tensorlistgetitem_scan_tensorarrayunstack_1_tensorlistfromtensor_0"─
_scan_while_tensorarrayv2read_2_tensorlistgetitem_scan_tensorarrayunstack_2_tensorlistfromtensorascan_while_tensorarrayv2read_2_tensorlistgetitem_scan_tensorarrayunstack_2_tensorlistfromtensor_0"╝
[scan_while_tensorarrayv2read_tensorlistgetitem_scan_tensorarrayunstack_tensorlistfromtensor]scan_while_tensorarrayv2read_tensorlistgetitem_scan_tensorarrayunstack_tensorlistfromtensor_0"\
+scan_while_unsortedsegmentsum_strided_slice-scan_while_unsortedsegmentsum_strided_slice_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : :         `: : : : : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         `:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
°
Х
,__inference_sequential_1_layer_call_fn_20090

inputs
unknown:`
	unknown_0:`
identityИвStatefulPartitionedCallщ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  `*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_15658|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  ``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:                  `: : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :                  `
 
_user_specified_nameinputs
Д
°
@__inference_dense_layer_call_and_return_conditional_losses_20239

inputs4
!tensordot_readvariableop_resource:	а`-
biasadd_readvariableop_resource:`
identityИвBiasAdd/ReadVariableOpвTensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	а`*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ┐
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ь
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Г
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*5
_output_shapes#
!:                  аК
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:                  К
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:`Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : з
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:М
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  `r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:`*
dtype0Е
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `l
IdentityIdentityBiasAdd:output:0^NoOp*
T0*4
_output_shapes"
 :                  `z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:                  а: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:] Y
5
_output_shapes#
!:                  а
 
_user_specified_nameinputs
█
Є
G__inference_sequential_1_layer_call_and_return_conditional_losses_15694
lambda_input)
layer_normalization_5_15688:`)
layer_normalization_5_15690:`
identityИв-layer_normalization_5/StatefulPartitionedCall╞
lambda/PartitionedCallPartitionedCalllambda_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  `* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_lambda_layer_call_and_return_conditional_losses_15461╟
-layer_normalization_5/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0layer_normalization_5_15688layer_normalization_5_15690*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  `*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_layer_normalization_5_layer_call_and_return_conditional_losses_15613Т
IdentityIdentity6layer_normalization_5/StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  `v
NoOpNoOp.^layer_normalization_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:                  `: : 2^
-layer_normalization_5/StatefulPartitionedCall-layer_normalization_5/StatefulPartitionedCall:b ^
4
_output_shapes"
 :                  `
&
_user_specified_namelambda_input
Й
Б
J__inference_node_prediction_layer_call_and_return_conditional_losses_16406

inputs3
!tensordot_readvariableop_resource:`-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвTensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:`*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ┐
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ь
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:В
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*4
_output_shapes"
 :                  `К
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:                  К
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         [
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : з
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:М
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Е
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  l
IdentityIdentityBiasAdd:output:0^NoOp*
T0*4
_output_shapes"
 :                  z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:                  `: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:\ X
4
_output_shapes"
 :                  `
 
_user_specified_nameinputs
№
╡
model_mpn_scan_while_cond_15136:
6model_mpn_scan_while_model_mpn_scan_while_loop_counter5
1model_mpn_scan_while_model_mpn_scan_strided_slice$
 model_mpn_scan_while_placeholder&
"model_mpn_scan_while_placeholder_1&
"model_mpn_scan_while_placeholder_2:
6model_mpn_scan_while_less_model_mpn_scan_strided_sliceQ
Mmodel_mpn_scan_while_model_mpn_scan_while_cond_15136___redundant_placeholder0Q
Mmodel_mpn_scan_while_model_mpn_scan_while_cond_15136___redundant_placeholder1Q
Mmodel_mpn_scan_while_model_mpn_scan_while_cond_15136___redundant_placeholder2Q
Mmodel_mpn_scan_while_model_mpn_scan_while_cond_15136___redundant_placeholder3!
model_mpn_scan_while_identity
Ь
model/mpn/scan/while/LessLess model_mpn_scan_while_placeholder6model_mpn_scan_while_less_model_mpn_scan_strided_slice*
T0*
_output_shapes
: п
model/mpn/scan/while/Less_1Less6model_mpn_scan_while_model_mpn_scan_while_loop_counter1model_mpn_scan_while_model_mpn_scan_strided_slice*
T0*
_output_shapes
: Е
model/mpn/scan/while/LogicalAnd
LogicalAndmodel/mpn/scan/while/Less_1:z:0model/mpn/scan/while/Less:z:0*
_output_shapes
: o
model/mpn/scan/while/IdentityIdentity#model/mpn/scan/while/LogicalAnd:z:0*
T0
*
_output_shapes
: "G
model_mpn_scan_while_identity&model/mpn/scan/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : :         `: : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         `:

_output_shapes
: :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
:
ъ5
╦
scan_while_body_19695&
"scan_while_scan_while_loop_counter!
scan_while_scan_strided_slice
scan_while_placeholder
scan_while_placeholder_1
scan_while_placeholder_2%
!scan_while_scan_strided_slice_1_0a
]scan_while_tensorarrayv2read_tensorlistgetitem_scan_tensorarrayunstack_tensorlistfromtensor_0e
ascan_while_tensorarrayv2read_1_tensorlistgetitem_scan_tensorarrayunstack_1_tensorlistfromtensor_0e
ascan_while_tensorarrayv2read_2_tensorlistgetitem_scan_tensorarrayunstack_2_tensorlistfromtensor_01
-scan_while_unsortedsegmentsum_strided_slice_0
scan_while_identity
scan_while_identity_1
scan_while_identity_2
scan_while_identity_3
scan_while_identity_4#
scan_while_scan_strided_slice_1_
[scan_while_tensorarrayv2read_tensorlistgetitem_scan_tensorarrayunstack_tensorlistfromtensorc
_scan_while_tensorarrayv2read_1_tensorlistgetitem_scan_tensorarrayunstack_1_tensorlistfromtensorc
_scan_while_tensorarrayv2read_2_tensorlistgetitem_scan_tensorarrayunstack_2_tensorlistfromtensor/
+scan_while_unsortedsegmentsum_strided_sliceН
<scan/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    `   ┐
.scan/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem]scan_while_tensorarrayv2read_tensorlistgetitem_scan_tensorarrayunstack_tensorlistfromtensor_0scan_while_placeholderEscan/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         `*
element_dtype0П
>scan/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╟
0scan/while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItemascan_while_tensorarrayv2read_1_tensorlistgetitem_scan_tensorarrayunstack_1_tensorlistfromtensor_0scan_while_placeholderGscan/while/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0П
>scan/while/TensorArrayV2Read_2/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╟
0scan/while/TensorArrayV2Read_2/TensorListGetItemTensorListGetItemascan_while_tensorarrayv2read_2_tensorlistgetitem_scan_tensorarrayunstack_2_tensorlistfromtensor_0scan_while_placeholderGscan/while/TensorArrayV2Read_2/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0o
scan/while/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       q
 scan/while/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       q
 scan/while/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ┴
scan/while/strided_sliceStridedSlice7scan/while/TensorArrayV2Read_2/TensorListGetItem:item:0'scan/while/strided_slice/stack:output:0)scan/while/strided_slice/stack_1:output:0)scan/while/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_maskб
scan/while/mulMul5scan/while/TensorArrayV2Read/TensorListGetItem:item:0!scan/while/strided_slice:output:0*
T0*'
_output_shapes
:         `q
 scan/while/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       s
"scan/while/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       s
"scan/while/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ▌
scan/while/strided_slice_1StridedSlice7scan/while/TensorArrayV2Read_1/TensorListGetItem:item:0)scan/while/strided_slice_1/stack:output:0+scan/while/strided_slice_1/stack_1:output:0+scan/while/strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask▌
scan/while/UnsortedSegmentSumUnsortedSegmentSumscan/while/mul:z:0#scan/while/strided_slice_1:output:0-scan_while_unsortedsegmentsum_strided_slice_0*
T0*
Tindices0*'
_output_shapes
:         `▐
/scan/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemscan_while_placeholder_2scan_while_placeholder&scan/while/UnsortedSegmentSum:output:0*
_output_shapes
: *
element_dtype0:щш╥R
scan/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :k
scan/while/addAddV2scan_while_placeholderscan/while/add/y:output:0*
T0*
_output_shapes
: T
scan/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :{
scan/while/add_1AddV2"scan_while_scan_while_loop_counterscan/while/add_1/y:output:0*
T0*
_output_shapes
: V
scan/while/IdentityIdentityscan/while/add_1:z:0*
T0*
_output_shapes
: a
scan/while/Identity_1Identityscan_while_scan_strided_slice*
T0*
_output_shapes
: V
scan/while/Identity_2Identityscan/while/add:z:0*
T0*
_output_shapes
: {
scan/while/Identity_3Identity&scan/while/UnsortedSegmentSum:output:0*
T0*'
_output_shapes
:         `Г
scan/while/Identity_4Identity?scan/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: "3
scan_while_identityscan/while/Identity:output:0"7
scan_while_identity_1scan/while/Identity_1:output:0"7
scan_while_identity_2scan/while/Identity_2:output:0"7
scan_while_identity_3scan/while/Identity_3:output:0"7
scan_while_identity_4scan/while/Identity_4:output:0"D
scan_while_scan_strided_slice_1!scan_while_scan_strided_slice_1_0"─
_scan_while_tensorarrayv2read_1_tensorlistgetitem_scan_tensorarrayunstack_1_tensorlistfromtensorascan_while_tensorarrayv2read_1_tensorlistgetitem_scan_tensorarrayunstack_1_tensorlistfromtensor_0"─
_scan_while_tensorarrayv2read_2_tensorlistgetitem_scan_tensorarrayunstack_2_tensorlistfromtensorascan_while_tensorarrayv2read_2_tensorlistgetitem_scan_tensorarrayunstack_2_tensorlistfromtensor_0"╝
[scan_while_tensorarrayv2read_tensorlistgetitem_scan_tensorarrayunstack_tensorlistfromtensor]scan_while_tensorarrayv2read_tensorlistgetitem_scan_tensorarrayunstack_tensorlistfromtensor_0"\
+scan_while_unsortedsegmentsum_strided_slice-scan_while_unsortedsegmentsum_strided_slice_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : :         `: : : : : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         `:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
Ы
ў
scan_while_cond_19412&
"scan_while_scan_while_loop_counter!
scan_while_scan_strided_slice
scan_while_placeholder
scan_while_placeholder_1
scan_while_placeholder_2&
"scan_while_less_scan_strided_slice=
9scan_while_scan_while_cond_19412___redundant_placeholder0=
9scan_while_scan_while_cond_19412___redundant_placeholder1=
9scan_while_scan_while_cond_19412___redundant_placeholder2=
9scan_while_scan_while_cond_19412___redundant_placeholder3
scan_while_identity
t
scan/while/LessLessscan_while_placeholder"scan_while_less_scan_strided_slice*
T0*
_output_shapes
: }
scan/while/Less_1Less"scan_while_scan_while_loop_counterscan_while_scan_strided_slice*
T0*
_output_shapes
: g
scan/while/LogicalAnd
LogicalAndscan/while/Less_1:z:0scan/while/Less:z:0*
_output_shapes
: [
scan/while/IdentityIdentityscan/while/LogicalAnd:z:0*
T0
*
_output_shapes
: "3
scan_while_identityscan/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : :         `: : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         `:

_output_shapes
: :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
:
а
ж
#__inference_mpn_layer_call_fn_19248
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
unknown:	а`
	unknown_0:`
	unknown_1:`
	unknown_2:`
	unknown_3:	└`
	unknown_4:`
	unknown_5:`
	unknown_6:`
identity

identity_1

identity_2

identity_3

identity_4ИвStatefulPartitionedCallу
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2*
Tout	
2*
_collective_manager_ids
 *╢
_output_shapesг
а:                  `:                  `:                  :                  :                  **
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8В *G
fBR@
>__inference_mpn_layer_call_and_return_conditional_losses_16354|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  `~

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*4
_output_shapes"
 :                  `~

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*4
_output_shapes"
 :                  ~

Identity_3Identity StatefulPartitionedCall:output:3^NoOp*
T0*4
_output_shapes"
 :                  ~

Identity_4Identity StatefulPartitionedCall:output:4^NoOp*
T0*4
_output_shapes"
 :                  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*┼
_input_shapes│
░:                  `:                  `:                  :                  :                  : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :                  `
"
_user_specified_name
inputs_0:^Z
4
_output_shapes"
 :                  `
"
_user_specified_name
inputs_1:^Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_2:^Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_3:^Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_4
ё
У
%__inference_dense_layer_call_fn_20209

inputs
unknown:	а`
	unknown_0:`
identityИвStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  `*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_15346|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  ``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:                  а: : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:                  а
 
_user_specified_nameinputs
Ъ
┼
E__inference_sequential_layer_call_and_return_conditional_losses_15504

inputs
dense_15492:	а`
dense_15494:`)
layer_normalization_4_15498:`)
layer_normalization_4_15500:`
identityИвdense/StatefulPartitionedCallв-layer_normalization_4/StatefulPartitionedCallю
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_15492dense_15494*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  `*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_15346р
lambda/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  `* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_lambda_layer_call_and_return_conditional_losses_15461╟
-layer_normalization_4/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0layer_normalization_4_15498layer_normalization_4_15500*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  `*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_layer_normalization_4_layer_call_and_return_conditional_losses_15413Т
IdentityIdentity6layer_normalization_4/StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  `Ц
NoOpNoOp^dense/StatefulPartitionedCall.^layer_normalization_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):                  а: : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2^
-layer_normalization_4/StatefulPartitionedCall-layer_normalization_4/StatefulPartitionedCall:] Y
5
_output_shapes#
!:                  а
 
_user_specified_nameinputs
К
Ы
,__inference_sequential_1_layer_call_fn_15674
lambda_input
unknown:`
	unknown_0:`
identityИвStatefulPartitionedCallя
StatefulPartitionedCallStatefulPartitionedCalllambda_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  `*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_15658|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  ``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:                  `: : 22
StatefulPartitionedCallStatefulPartitionedCall:b ^
4
_output_shapes"
 :                  `
&
_user_specified_namelambda_input
╝	
]
A__inference_lambda_layer_call_and_return_conditional_losses_18862

inputs
identityO

Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?k
Gelu/mulMulGelu/mul/x:output:0inputs*
T0*4
_output_shapes"
 :                  @P
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *є╡?t
Gelu/truedivRealDivinputsGelu/Cast/x:output:0*
T0*4
_output_shapes"
 :                  @`
Gelu/ErfErfGelu/truediv:z:0*
T0*4
_output_shapes"
 :                  @O

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?s
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*4
_output_shapes"
 :                  @l

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*4
_output_shapes"
 :                  @c
IdentityIdentityGelu/mul_1:z:0*
T0*4
_output_shapes"
 :                  @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :                  @:\ X
4
_output_shapes"
 :                  @
 
_user_specified_nameinputs
й
├
mpn_scan_while_cond_18594.
*mpn_scan_while_mpn_scan_while_loop_counter)
%mpn_scan_while_mpn_scan_strided_slice
mpn_scan_while_placeholder 
mpn_scan_while_placeholder_1 
mpn_scan_while_placeholder_2.
*mpn_scan_while_less_mpn_scan_strided_sliceE
Ampn_scan_while_mpn_scan_while_cond_18594___redundant_placeholder0E
Ampn_scan_while_mpn_scan_while_cond_18594___redundant_placeholder1E
Ampn_scan_while_mpn_scan_while_cond_18594___redundant_placeholder2E
Ampn_scan_while_mpn_scan_while_cond_18594___redundant_placeholder3
mpn_scan_while_identity
Д
mpn/scan/while/LessLessmpn_scan_while_placeholder*mpn_scan_while_less_mpn_scan_strided_slice*
T0*
_output_shapes
: С
mpn/scan/while/Less_1Less*mpn_scan_while_mpn_scan_while_loop_counter%mpn_scan_while_mpn_scan_strided_slice*
T0*
_output_shapes
: s
mpn/scan/while/LogicalAnd
LogicalAndmpn/scan/while/Less_1:z:0mpn/scan/while/Less:z:0*
_output_shapes
: c
mpn/scan/while/IdentityIdentitympn/scan/while/LogicalAnd:z:0*
T0
*
_output_shapes
: ";
mpn_scan_while_identity mpn/scan/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : :         `: : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         `:

_output_shapes
: :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
:
К
Ю
5__inference_layer_normalization_1_layer_call_fn_19017

inputs
unknown:@
	unknown_0:@
identityИвStatefulPartitionedCallЄ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_layer_normalization_1_layer_call_and_return_conditional_losses_15930|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:                  @: : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :                  @
 
_user_specified_nameinputs
уS
Ы
@__inference_model_layer_call_and_return_conditional_losses_16413

inputs
inputs_1
inputs_2
inputs_3!
node_ide1_15738:@
node_ide1_15740:@'
layer_normalization_15804:@'
layer_normalization_15806:@!
edge_ide1_15840:@
edge_ide1_15842:@!
node_ide2_15876:@`
node_ide2_15878:`)
layer_normalization_1_15931:@)
layer_normalization_1_15933:@!
edge_ide2_15967:@`
edge_ide2_15969:`)
layer_normalization_2_16025:`)
layer_normalization_2_16027:`)
layer_normalization_3_16078:`)
layer_normalization_3_16080:`
	mpn_16355:	а`
	mpn_16357:`
	mpn_16359:`
	mpn_16361:`
	mpn_16363:	└`
	mpn_16365:`
	mpn_16367:`
	mpn_16369:`'
node_prediction_16407:`#
node_prediction_16409:
identityИв!edge_ide1/StatefulPartitionedCallв!edge_ide2/StatefulPartitionedCallв+layer_normalization/StatefulPartitionedCallв-layer_normalization_1/StatefulPartitionedCallв-layer_normalization_2/StatefulPartitionedCallв-layer_normalization_3/StatefulPartitionedCallвmpn/StatefulPartitionedCallв!node_ide1/StatefulPartitionedCallв!node_ide2/StatefulPartitionedCallв'node_prediction/StatefulPartitionedCall■
!node_ide1/StatefulPartitionedCallStatefulPartitionedCallinputsnode_ide1_15738node_ide1_15740*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_node_ide1_layer_call_and_return_conditional_losses_15737ф
lambda/PartitionedCallPartitionedCall*node_ide1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_lambda_layer_call_and_return_conditional_losses_15754┐
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0layer_normalization_15804layer_normalization_15806*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *W
fRRP
N__inference_layer_normalization_layer_call_and_return_conditional_losses_15803А
!edge_ide1/StatefulPartitionedCallStatefulPartitionedCallinputs_1edge_ide1_15840edge_ide1_15842*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_edge_ide1_layer_call_and_return_conditional_losses_15839м
!node_ide2/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0node_ide2_15876node_ide2_15878*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  `*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_node_ide2_layer_call_and_return_conditional_losses_15875ц
lambda/PartitionedCall_1PartitionedCall*edge_ide1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_lambda_layer_call_and_return_conditional_losses_15754ц
lambda/PartitionedCall_2PartitionedCall*node_ide2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  `* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_lambda_layer_call_and_return_conditional_losses_15364╔
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCall!lambda/PartitionedCall_1:output:0layer_normalization_1_15931layer_normalization_1_15933*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_layer_normalization_1_layer_call_and_return_conditional_losses_15930о
!edge_ide2/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_1/StatefulPartitionedCall:output:0edge_ide2_15967edge_ide2_15969*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  `*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_edge_ide2_layer_call_and_return_conditional_losses_15966}
,tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
.tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
.tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╩
&tf.__operators__.getitem/strided_sliceStridedSliceinputs_15tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*4
_output_shapes"
 :                  *
ellipsis_maskц
lambda/PartitionedCall_3PartitionedCall*edge_ide2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  `* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_lambda_layer_call_and_return_conditional_losses_15364╔
-layer_normalization_2/StatefulPartitionedCallStatefulPartitionedCall!lambda/PartitionedCall_2:output:0layer_normalization_2_16025layer_normalization_2_16027*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  `*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_layer_normalization_2_layer_call_and_return_conditional_losses_16024╔
-layer_normalization_3/StatefulPartitionedCallStatefulPartitionedCall!lambda/PartitionedCall_3:output:0layer_normalization_3_16078layer_normalization_3_16080*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  `*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_layer_normalization_3_layer_call_and_return_conditional_losses_16077{
tf.ones_like/ones_like/ShapeShape/tf.__operators__.getitem/strided_slice:output:0*
T0*
_output_shapes
:a
tf.ones_like/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?л
tf.ones_like/ones_likeFill%tf.ones_like/ones_like/Shape:output:0%tf.ones_like/ones_like/Const:output:0*
T0*4
_output_shapes"
 :                  ▄
mpn/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_2/StatefulPartitionedCall:output:06layer_normalization_3/StatefulPartitionedCall:output:0inputs_2tf.ones_like/ones_like:output:0inputs_3	mpn_16355	mpn_16357	mpn_16359	mpn_16361	mpn_16363	mpn_16365	mpn_16367	mpn_16369*
Tin
2*
Tout	
2*
_collective_manager_ids
 *╢
_output_shapesг
а:                  `:                  `:                  :                  :                  **
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8В *G
fBR@
>__inference_mpn_layer_call_and_return_conditional_losses_16354┤
'node_prediction/StatefulPartitionedCallStatefulPartitionedCall$mpn/StatefulPartitionedCall:output:0node_prediction_16407node_prediction_16409*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_node_prediction_layer_call_and_return_conditional_losses_16406М
IdentityIdentity0node_prediction/StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  ▄
NoOpNoOp"^edge_ide1/StatefulPartitionedCall"^edge_ide2/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall.^layer_normalization_1/StatefulPartitionedCall.^layer_normalization_2/StatefulPartitionedCall.^layer_normalization_3/StatefulPartitionedCall^mpn/StatefulPartitionedCall"^node_ide1/StatefulPartitionedCall"^node_ide2/StatefulPartitionedCall(^node_prediction/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*╔
_input_shapes╖
┤:                  :                  :                  :                  : : : : : : : : : : : : : : : : : : : : : : : : : : 2F
!edge_ide1/StatefulPartitionedCall!edge_ide1/StatefulPartitionedCall2F
!edge_ide2/StatefulPartitionedCall!edge_ide2/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2^
-layer_normalization_1/StatefulPartitionedCall-layer_normalization_1/StatefulPartitionedCall2^
-layer_normalization_2/StatefulPartitionedCall-layer_normalization_2/StatefulPartitionedCall2^
-layer_normalization_3/StatefulPartitionedCall-layer_normalization_3/StatefulPartitionedCall2:
mpn/StatefulPartitionedCallmpn/StatefulPartitionedCall2F
!node_ide1/StatefulPartitionedCall!node_ide1/StatefulPartitionedCall2F
!node_ide2/StatefulPartitionedCall!node_ide2/StatefulPartitionedCall2R
'node_prediction/StatefulPartitionedCall'node_prediction/StatefulPartitionedCall:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs:\X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs:\X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs:\X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
╝	
]
A__inference_lambda_layer_call_and_return_conditional_losses_16872

inputs
identityO

Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?k
Gelu/mulMulGelu/mul/x:output:0inputs*
T0*4
_output_shapes"
 :                  @P
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *є╡?t
Gelu/truedivRealDivinputsGelu/Cast/x:output:0*
T0*4
_output_shapes"
 :                  @`
Gelu/ErfErfGelu/truediv:z:0*
T0*4
_output_shapes"
 :                  @O

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?s
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*4
_output_shapes"
 :                  @l

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*4
_output_shapes"
 :                  @c
IdentityIdentityGelu/mul_1:z:0*
T0*4
_output_shapes"
 :                  @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :                  @:\ X
4
_output_shapes"
 :                  @
 
_user_specified_nameinputs
К
Ы
,__inference_sequential_1_layer_call_fn_15627
lambda_input
unknown:`
	unknown_0:`
identityИвStatefulPartitionedCallя
StatefulPartitionedCallStatefulPartitionedCalllambda_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  `*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_15620|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  ``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:                  `: : 22
StatefulPartitionedCallStatefulPartitionedCall:b ^
4
_output_shapes"
 :                  `
&
_user_specified_namelambda_input
бс
╨6
!__inference__traced_restore_20891
file_prefix3
!assignvariableop_node_ide1_kernel:@/
!assignvariableop_1_node_ide1_bias:@:
,assignvariableop_2_layer_normalization_gamma:@9
+assignvariableop_3_layer_normalization_beta:@5
#assignvariableop_4_node_ide2_kernel:@`/
!assignvariableop_5_node_ide2_bias:`5
#assignvariableop_6_edge_ide1_kernel:@/
!assignvariableop_7_edge_ide1_bias:@<
.assignvariableop_8_layer_normalization_1_gamma:@;
-assignvariableop_9_layer_normalization_1_beta:@6
$assignvariableop_10_edge_ide2_kernel:@`0
"assignvariableop_11_edge_ide2_bias:`=
/assignvariableop_12_layer_normalization_2_gamma:`<
.assignvariableop_13_layer_normalization_2_beta:`=
/assignvariableop_14_layer_normalization_3_gamma:`<
.assignvariableop_15_layer_normalization_3_beta:`<
*assignvariableop_16_node_prediction_kernel:`6
(assignvariableop_17_node_prediction_bias:3
 assignvariableop_18_dense_kernel:	а`,
assignvariableop_19_dense_bias:`=
/assignvariableop_20_layer_normalization_4_gamma:`<
.assignvariableop_21_layer_normalization_4_beta:`9
&assignvariableop_22_mpn_dense_1_kernel:	└`2
$assignvariableop_23_mpn_dense_1_bias:`=
/assignvariableop_24_layer_normalization_5_gamma:`<
.assignvariableop_25_layer_normalization_5_beta:`'
assignvariableop_26_iteration:	 +
!assignvariableop_27_learning_rate: =
+assignvariableop_28_adam_m_node_ide1_kernel:@=
+assignvariableop_29_adam_v_node_ide1_kernel:@7
)assignvariableop_30_adam_m_node_ide1_bias:@7
)assignvariableop_31_adam_v_node_ide1_bias:@B
4assignvariableop_32_adam_m_layer_normalization_gamma:@B
4assignvariableop_33_adam_v_layer_normalization_gamma:@A
3assignvariableop_34_adam_m_layer_normalization_beta:@A
3assignvariableop_35_adam_v_layer_normalization_beta:@=
+assignvariableop_36_adam_m_node_ide2_kernel:@`=
+assignvariableop_37_adam_v_node_ide2_kernel:@`7
)assignvariableop_38_adam_m_node_ide2_bias:`7
)assignvariableop_39_adam_v_node_ide2_bias:`=
+assignvariableop_40_adam_m_edge_ide1_kernel:@=
+assignvariableop_41_adam_v_edge_ide1_kernel:@7
)assignvariableop_42_adam_m_edge_ide1_bias:@7
)assignvariableop_43_adam_v_edge_ide1_bias:@D
6assignvariableop_44_adam_m_layer_normalization_1_gamma:@D
6assignvariableop_45_adam_v_layer_normalization_1_gamma:@C
5assignvariableop_46_adam_m_layer_normalization_1_beta:@C
5assignvariableop_47_adam_v_layer_normalization_1_beta:@=
+assignvariableop_48_adam_m_edge_ide2_kernel:@`=
+assignvariableop_49_adam_v_edge_ide2_kernel:@`7
)assignvariableop_50_adam_m_edge_ide2_bias:`7
)assignvariableop_51_adam_v_edge_ide2_bias:`D
6assignvariableop_52_adam_m_layer_normalization_2_gamma:`D
6assignvariableop_53_adam_v_layer_normalization_2_gamma:`C
5assignvariableop_54_adam_m_layer_normalization_2_beta:`C
5assignvariableop_55_adam_v_layer_normalization_2_beta:`D
6assignvariableop_56_adam_m_layer_normalization_3_gamma:`D
6assignvariableop_57_adam_v_layer_normalization_3_gamma:`C
5assignvariableop_58_adam_m_layer_normalization_3_beta:`C
5assignvariableop_59_adam_v_layer_normalization_3_beta:`:
'assignvariableop_60_adam_m_dense_kernel:	а`:
'assignvariableop_61_adam_v_dense_kernel:	а`3
%assignvariableop_62_adam_m_dense_bias:`3
%assignvariableop_63_adam_v_dense_bias:`D
6assignvariableop_64_adam_m_layer_normalization_4_gamma:`D
6assignvariableop_65_adam_v_layer_normalization_4_gamma:`C
5assignvariableop_66_adam_m_layer_normalization_4_beta:`C
5assignvariableop_67_adam_v_layer_normalization_4_beta:`@
-assignvariableop_68_adam_m_mpn_dense_1_kernel:	└`@
-assignvariableop_69_adam_v_mpn_dense_1_kernel:	└`9
+assignvariableop_70_adam_m_mpn_dense_1_bias:`9
+assignvariableop_71_adam_v_mpn_dense_1_bias:`D
6assignvariableop_72_adam_m_layer_normalization_5_gamma:`D
6assignvariableop_73_adam_v_layer_normalization_5_gamma:`C
5assignvariableop_74_adam_m_layer_normalization_5_beta:`C
5assignvariableop_75_adam_v_layer_normalization_5_beta:`C
1assignvariableop_76_adam_m_node_prediction_kernel:`C
1assignvariableop_77_adam_v_node_prediction_kernel:`=
/assignvariableop_78_adam_m_node_prediction_bias:=
/assignvariableop_79_adam_v_node_prediction_bias:%
assignvariableop_80_total_1: %
assignvariableop_81_count_1: #
assignvariableop_82_total: #
assignvariableop_83_count: 
identity_85ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_28вAssignVariableOp_29вAssignVariableOp_3вAssignVariableOp_30вAssignVariableOp_31вAssignVariableOp_32вAssignVariableOp_33вAssignVariableOp_34вAssignVariableOp_35вAssignVariableOp_36вAssignVariableOp_37вAssignVariableOp_38вAssignVariableOp_39вAssignVariableOp_4вAssignVariableOp_40вAssignVariableOp_41вAssignVariableOp_42вAssignVariableOp_43вAssignVariableOp_44вAssignVariableOp_45вAssignVariableOp_46вAssignVariableOp_47вAssignVariableOp_48вAssignVariableOp_49вAssignVariableOp_5вAssignVariableOp_50вAssignVariableOp_51вAssignVariableOp_52вAssignVariableOp_53вAssignVariableOp_54вAssignVariableOp_55вAssignVariableOp_56вAssignVariableOp_57вAssignVariableOp_58вAssignVariableOp_59вAssignVariableOp_6вAssignVariableOp_60вAssignVariableOp_61вAssignVariableOp_62вAssignVariableOp_63вAssignVariableOp_64вAssignVariableOp_65вAssignVariableOp_66вAssignVariableOp_67вAssignVariableOp_68вAssignVariableOp_69вAssignVariableOp_7вAssignVariableOp_70вAssignVariableOp_71вAssignVariableOp_72вAssignVariableOp_73вAssignVariableOp_74вAssignVariableOp_75вAssignVariableOp_76вAssignVariableOp_77вAssignVariableOp_78вAssignVariableOp_79вAssignVariableOp_8вAssignVariableOp_80вAssignVariableOp_81вAssignVariableOp_82вAssignVariableOp_83вAssignVariableOp_9ї"
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:U*
dtype0*Ы"
valueС"BО"UB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/49/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/50/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/51/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/52/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЭ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:U*
dtype0*┐
value╡B▓UB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ╩
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ъ
_output_shapes╫
╘:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*c
dtypesY
W2U	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:┤
AssignVariableOpAssignVariableOp!assignvariableop_node_ide1_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:╕
AssignVariableOp_1AssignVariableOp!assignvariableop_1_node_ide1_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_2AssignVariableOp,assignvariableop_2_layer_normalization_gammaIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_3AssignVariableOp+assignvariableop_3_layer_normalization_betaIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:║
AssignVariableOp_4AssignVariableOp#assignvariableop_4_node_ide2_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:╕
AssignVariableOp_5AssignVariableOp!assignvariableop_5_node_ide2_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:║
AssignVariableOp_6AssignVariableOp#assignvariableop_6_edge_ide1_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:╕
AssignVariableOp_7AssignVariableOp!assignvariableop_7_edge_ide1_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:┼
AssignVariableOp_8AssignVariableOp.assignvariableop_8_layer_normalization_1_gammaIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:─
AssignVariableOp_9AssignVariableOp-assignvariableop_9_layer_normalization_1_betaIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:╜
AssignVariableOp_10AssignVariableOp$assignvariableop_10_edge_ide2_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:╗
AssignVariableOp_11AssignVariableOp"assignvariableop_11_edge_ide2_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:╚
AssignVariableOp_12AssignVariableOp/assignvariableop_12_layer_normalization_2_gammaIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:╟
AssignVariableOp_13AssignVariableOp.assignvariableop_13_layer_normalization_2_betaIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:╚
AssignVariableOp_14AssignVariableOp/assignvariableop_14_layer_normalization_3_gammaIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:╟
AssignVariableOp_15AssignVariableOp.assignvariableop_15_layer_normalization_3_betaIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_16AssignVariableOp*assignvariableop_16_node_prediction_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_17AssignVariableOp(assignvariableop_17_node_prediction_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:╣
AssignVariableOp_18AssignVariableOp assignvariableop_18_dense_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:╖
AssignVariableOp_19AssignVariableOpassignvariableop_19_dense_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:╚
AssignVariableOp_20AssignVariableOp/assignvariableop_20_layer_normalization_4_gammaIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:╟
AssignVariableOp_21AssignVariableOp.assignvariableop_21_layer_normalization_4_betaIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:┐
AssignVariableOp_22AssignVariableOp&assignvariableop_22_mpn_dense_1_kernelIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:╜
AssignVariableOp_23AssignVariableOp$assignvariableop_23_mpn_dense_1_biasIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:╚
AssignVariableOp_24AssignVariableOp/assignvariableop_24_layer_normalization_5_gammaIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:╟
AssignVariableOp_25AssignVariableOp.assignvariableop_25_layer_normalization_5_betaIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0	*
_output_shapes
:╢
AssignVariableOp_26AssignVariableOpassignvariableop_26_iterationIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:║
AssignVariableOp_27AssignVariableOp!assignvariableop_27_learning_rateIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:─
AssignVariableOp_28AssignVariableOp+assignvariableop_28_adam_m_node_ide1_kernelIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:─
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_v_node_ide1_kernelIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_m_node_ide1_biasIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_31AssignVariableOp)assignvariableop_31_adam_v_node_ide1_biasIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:═
AssignVariableOp_32AssignVariableOp4assignvariableop_32_adam_m_layer_normalization_gammaIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:═
AssignVariableOp_33AssignVariableOp4assignvariableop_33_adam_v_layer_normalization_gammaIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:╠
AssignVariableOp_34AssignVariableOp3assignvariableop_34_adam_m_layer_normalization_betaIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:╠
AssignVariableOp_35AssignVariableOp3assignvariableop_35_adam_v_layer_normalization_betaIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:─
AssignVariableOp_36AssignVariableOp+assignvariableop_36_adam_m_node_ide2_kernelIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:─
AssignVariableOp_37AssignVariableOp+assignvariableop_37_adam_v_node_ide2_kernelIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_38AssignVariableOp)assignvariableop_38_adam_m_node_ide2_biasIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_39AssignVariableOp)assignvariableop_39_adam_v_node_ide2_biasIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:─
AssignVariableOp_40AssignVariableOp+assignvariableop_40_adam_m_edge_ide1_kernelIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:─
AssignVariableOp_41AssignVariableOp+assignvariableop_41_adam_v_edge_ide1_kernelIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_42AssignVariableOp)assignvariableop_42_adam_m_edge_ide1_biasIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_43AssignVariableOp)assignvariableop_43_adam_v_edge_ide1_biasIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:╧
AssignVariableOp_44AssignVariableOp6assignvariableop_44_adam_m_layer_normalization_1_gammaIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:╧
AssignVariableOp_45AssignVariableOp6assignvariableop_45_adam_v_layer_normalization_1_gammaIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:╬
AssignVariableOp_46AssignVariableOp5assignvariableop_46_adam_m_layer_normalization_1_betaIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:╬
AssignVariableOp_47AssignVariableOp5assignvariableop_47_adam_v_layer_normalization_1_betaIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:─
AssignVariableOp_48AssignVariableOp+assignvariableop_48_adam_m_edge_ide2_kernelIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:─
AssignVariableOp_49AssignVariableOp+assignvariableop_49_adam_v_edge_ide2_kernelIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_50AssignVariableOp)assignvariableop_50_adam_m_edge_ide2_biasIdentity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_51AssignVariableOp)assignvariableop_51_adam_v_edge_ide2_biasIdentity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:╧
AssignVariableOp_52AssignVariableOp6assignvariableop_52_adam_m_layer_normalization_2_gammaIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:╧
AssignVariableOp_53AssignVariableOp6assignvariableop_53_adam_v_layer_normalization_2_gammaIdentity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:╬
AssignVariableOp_54AssignVariableOp5assignvariableop_54_adam_m_layer_normalization_2_betaIdentity_54:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:╬
AssignVariableOp_55AssignVariableOp5assignvariableop_55_adam_v_layer_normalization_2_betaIdentity_55:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:╧
AssignVariableOp_56AssignVariableOp6assignvariableop_56_adam_m_layer_normalization_3_gammaIdentity_56:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:╧
AssignVariableOp_57AssignVariableOp6assignvariableop_57_adam_v_layer_normalization_3_gammaIdentity_57:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:╬
AssignVariableOp_58AssignVariableOp5assignvariableop_58_adam_m_layer_normalization_3_betaIdentity_58:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:╬
AssignVariableOp_59AssignVariableOp5assignvariableop_59_adam_v_layer_normalization_3_betaIdentity_59:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_60AssignVariableOp'assignvariableop_60_adam_m_dense_kernelIdentity_60:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_61AssignVariableOp'assignvariableop_61_adam_v_dense_kernelIdentity_61:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:╛
AssignVariableOp_62AssignVariableOp%assignvariableop_62_adam_m_dense_biasIdentity_62:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:╛
AssignVariableOp_63AssignVariableOp%assignvariableop_63_adam_v_dense_biasIdentity_63:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:╧
AssignVariableOp_64AssignVariableOp6assignvariableop_64_adam_m_layer_normalization_4_gammaIdentity_64:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:╧
AssignVariableOp_65AssignVariableOp6assignvariableop_65_adam_v_layer_normalization_4_gammaIdentity_65:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:╬
AssignVariableOp_66AssignVariableOp5assignvariableop_66_adam_m_layer_normalization_4_betaIdentity_66:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:╬
AssignVariableOp_67AssignVariableOp5assignvariableop_67_adam_v_layer_normalization_4_betaIdentity_67:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:╞
AssignVariableOp_68AssignVariableOp-assignvariableop_68_adam_m_mpn_dense_1_kernelIdentity_68:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:╞
AssignVariableOp_69AssignVariableOp-assignvariableop_69_adam_v_mpn_dense_1_kernelIdentity_69:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:─
AssignVariableOp_70AssignVariableOp+assignvariableop_70_adam_m_mpn_dense_1_biasIdentity_70:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:─
AssignVariableOp_71AssignVariableOp+assignvariableop_71_adam_v_mpn_dense_1_biasIdentity_71:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:╧
AssignVariableOp_72AssignVariableOp6assignvariableop_72_adam_m_layer_normalization_5_gammaIdentity_72:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:╧
AssignVariableOp_73AssignVariableOp6assignvariableop_73_adam_v_layer_normalization_5_gammaIdentity_73:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:╬
AssignVariableOp_74AssignVariableOp5assignvariableop_74_adam_m_layer_normalization_5_betaIdentity_74:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:╬
AssignVariableOp_75AssignVariableOp5assignvariableop_75_adam_v_layer_normalization_5_betaIdentity_75:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:╩
AssignVariableOp_76AssignVariableOp1assignvariableop_76_adam_m_node_prediction_kernelIdentity_76:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:╩
AssignVariableOp_77AssignVariableOp1assignvariableop_77_adam_v_node_prediction_kernelIdentity_77:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:╚
AssignVariableOp_78AssignVariableOp/assignvariableop_78_adam_m_node_prediction_biasIdentity_78:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:╚
AssignVariableOp_79AssignVariableOp/assignvariableop_79_adam_v_node_prediction_biasIdentity_79:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:┤
AssignVariableOp_80AssignVariableOpassignvariableop_80_total_1Identity_80:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:┤
AssignVariableOp_81AssignVariableOpassignvariableop_81_count_1Identity_81:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:▓
AssignVariableOp_82AssignVariableOpassignvariableop_82_totalIdentity_82:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:▓
AssignVariableOp_83AssignVariableOpassignvariableop_83_countIdentity_83:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 З
Identity_84Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_85IdentityIdentity_84:output:0^NoOp_1*
T0*
_output_shapes
: Ї
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_85Identity_85:output:0*┐
_input_shapesн
к: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Р%
є
P__inference_layer_normalization_2_layer_call_and_return_conditional_losses_16024

inputs+
mul_3_readvariableop_resource:`)
add_readvariableop_resource:`
identityИвadd/ReadVariableOpвmul_3/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/xConst*
_output_shapes
: *
dtype0*
value	B :S
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: _
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
mul_1Mulmul:z:0strided_slice_1:output:0*
T0*
_output_shapes
: _
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_2Mulmul_2/x:output:0strided_slice_2:output:0*
T0*
_output_shapes
: Q
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Н
Reshape/shapePackReshape/shape/0:output:0	mul_1:z:0	mul_2:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:         `L
ones/packedPack	mul_1:z:0*
N*
T0*
_output_shapes
:O

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?e
onesFillones/packed:output:0ones/Const:output:0*
T0*#
_output_shapes
:         M
zeros/packedPack	mul_1:z:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    h
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*#
_output_shapes
:         H
ConstConst*
_output_shapes
: *
dtype0*
valueB J
Const_1Const*
_output_shapes
: *
dtype0*
valueB г
FusedBatchNormV3FusedBatchNormV3Reshape:output:0ones:output:0zeros:output:0Const:output:0Const_1:output:0*
T0*
U0*o
_output_shapes]
[:         `:         :         :         :         :*
data_formatNCHW*
epsilon%oГ:y
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*4
_output_shapes"
 :                  `n
mul_3/ReadVariableOpReadVariableOpmul_3_readvariableop_resource*
_output_shapes
:`*
dtype0}
mul_3MulReshape_1:output:0mul_3/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:`*
dtype0r
addAddV2	mul_3:z:0add/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `c
IdentityIdentityadd:z:0^NoOp*
T0*4
_output_shapes"
 :                  `r
NoOpNoOp^add/ReadVariableOp^mul_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:                  `: : 2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_3/ReadVariableOpmul_3/ReadVariableOp:\ X
4
_output_shapes"
 :                  `
 
_user_specified_nameinputs
О%
ё
N__inference_layer_normalization_layer_call_and_return_conditional_losses_15803

inputs+
mul_3_readvariableop_resource:@)
add_readvariableop_resource:@
identityИвadd/ReadVariableOpвmul_3/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/xConst*
_output_shapes
: *
dtype0*
value	B :S
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: _
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
mul_1Mulmul:z:0strided_slice_1:output:0*
T0*
_output_shapes
: _
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_2Mulmul_2/x:output:0strided_slice_2:output:0*
T0*
_output_shapes
: Q
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Н
Reshape/shapePackReshape/shape/0:output:0	mul_1:z:0	mul_2:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:         @L
ones/packedPack	mul_1:z:0*
N*
T0*
_output_shapes
:O

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?e
onesFillones/packed:output:0ones/Const:output:0*
T0*#
_output_shapes
:         M
zeros/packedPack	mul_1:z:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    h
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*#
_output_shapes
:         H
ConstConst*
_output_shapes
: *
dtype0*
valueB J
Const_1Const*
_output_shapes
: *
dtype0*
valueB г
FusedBatchNormV3FusedBatchNormV3Reshape:output:0ones:output:0zeros:output:0Const:output:0Const_1:output:0*
T0*
U0*o
_output_shapes]
[:         @:         :         :         :         :*
data_formatNCHW*
epsilon%oГ:y
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*4
_output_shapes"
 :                  @n
mul_3/ReadVariableOpReadVariableOpmul_3_readvariableop_resource*
_output_shapes
:@*
dtype0}
mul_3MulReshape_1:output:0mul_3/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  @j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:@*
dtype0r
addAddV2	mul_3:z:0add/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  @c
IdentityIdentityadd:z:0^NoOp*
T0*4
_output_shapes"
 :                  @r
NoOpNoOp^add/ReadVariableOp^mul_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:                  @: : 2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_3/ReadVariableOpmul_3/ReadVariableOp:\ X
4
_output_shapes"
 :                  @
 
_user_specified_nameinputs
▀;
┬
G__inference_sequential_1_layer_call_and_return_conditional_losses_20145

inputsA
3layer_normalization_5_mul_3_readvariableop_resource:`?
1layer_normalization_5_add_readvariableop_resource:`
identityИв(layer_normalization_5/add/ReadVariableOpв*layer_normalization_5/mul_3/ReadVariableOpV
lambda/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?y
lambda/Gelu/mulMullambda/Gelu/mul/x:output:0inputs*
T0*4
_output_shapes"
 :                  `W
lambda/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *є╡?В
lambda/Gelu/truedivRealDivinputslambda/Gelu/Cast/x:output:0*
T0*4
_output_shapes"
 :                  `n
lambda/Gelu/ErfErflambda/Gelu/truediv:z:0*
T0*4
_output_shapes"
 :                  `V
lambda/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?И
lambda/Gelu/addAddV2lambda/Gelu/add/x:output:0lambda/Gelu/Erf:y:0*
T0*4
_output_shapes"
 :                  `Б
lambda/Gelu/mul_1Mullambda/Gelu/mul:z:0lambda/Gelu/add:z:0*
T0*4
_output_shapes"
 :                  ``
layer_normalization_5/ShapeShapelambda/Gelu/mul_1:z:0*
T0*
_output_shapes
:s
)layer_normalization_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+layer_normalization_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┐
#layer_normalization_5/strided_sliceStridedSlice$layer_normalization_5/Shape:output:02layer_normalization_5/strided_slice/stack:output:04layer_normalization_5/strided_slice/stack_1:output:04layer_normalization_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization_5/mul/xConst*
_output_shapes
: *
dtype0*
value	B :Х
layer_normalization_5/mulMul$layer_normalization_5/mul/x:output:0,layer_normalization_5/strided_slice:output:0*
T0*
_output_shapes
: u
+layer_normalization_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╟
%layer_normalization_5/strided_slice_1StridedSlice$layer_normalization_5/Shape:output:04layer_normalization_5/strided_slice_1/stack:output:06layer_normalization_5/strided_slice_1/stack_1:output:06layer_normalization_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskТ
layer_normalization_5/mul_1Mullayer_normalization_5/mul:z:0.layer_normalization_5/strided_slice_1:output:0*
T0*
_output_shapes
: u
+layer_normalization_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╟
%layer_normalization_5/strided_slice_2StridedSlice$layer_normalization_5/Shape:output:04layer_normalization_5/strided_slice_2/stack:output:06layer_normalization_5/strided_slice_2/stack_1:output:06layer_normalization_5/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
layer_normalization_5/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :Ы
layer_normalization_5/mul_2Mul&layer_normalization_5/mul_2/x:output:0.layer_normalization_5/strided_slice_2:output:0*
T0*
_output_shapes
: g
%layer_normalization_5/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :g
%layer_normalization_5/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :√
#layer_normalization_5/Reshape/shapePack.layer_normalization_5/Reshape/shape/0:output:0layer_normalization_5/mul_1:z:0layer_normalization_5/mul_2:z:0.layer_normalization_5/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:з
layer_normalization_5/ReshapeReshapelambda/Gelu/mul_1:z:0,layer_normalization_5/Reshape/shape:output:0*
T0*/
_output_shapes
:         `x
!layer_normalization_5/ones/packedPacklayer_normalization_5/mul_1:z:0*
N*
T0*
_output_shapes
:e
 layer_normalization_5/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?з
layer_normalization_5/onesFill*layer_normalization_5/ones/packed:output:0)layer_normalization_5/ones/Const:output:0*
T0*#
_output_shapes
:         y
"layer_normalization_5/zeros/packedPacklayer_normalization_5/mul_1:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization_5/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    к
layer_normalization_5/zerosFill+layer_normalization_5/zeros/packed:output:0*layer_normalization_5/zeros/Const:output:0*
T0*#
_output_shapes
:         ^
layer_normalization_5/ConstConst*
_output_shapes
: *
dtype0*
valueB `
layer_normalization_5/Const_1Const*
_output_shapes
: *
dtype0*
valueB з
&layer_normalization_5/FusedBatchNormV3FusedBatchNormV3&layer_normalization_5/Reshape:output:0#layer_normalization_5/ones:output:0$layer_normalization_5/zeros:output:0$layer_normalization_5/Const:output:0&layer_normalization_5/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:         `:         :         :         :         :*
data_formatNCHW*
epsilon%oГ:╗
layer_normalization_5/Reshape_1Reshape*layer_normalization_5/FusedBatchNormV3:y:0$layer_normalization_5/Shape:output:0*
T0*4
_output_shapes"
 :                  `Ъ
*layer_normalization_5/mul_3/ReadVariableOpReadVariableOp3layer_normalization_5_mul_3_readvariableop_resource*
_output_shapes
:`*
dtype0┐
layer_normalization_5/mul_3Mul(layer_normalization_5/Reshape_1:output:02layer_normalization_5/mul_3/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `Ц
(layer_normalization_5/add/ReadVariableOpReadVariableOp1layer_normalization_5_add_readvariableop_resource*
_output_shapes
:`*
dtype0┤
layer_normalization_5/addAddV2layer_normalization_5/mul_3:z:00layer_normalization_5/add/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `y
IdentityIdentitylayer_normalization_5/add:z:0^NoOp*
T0*4
_output_shapes"
 :                  `Ю
NoOpNoOp)^layer_normalization_5/add/ReadVariableOp+^layer_normalization_5/mul_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:                  `: : 2T
(layer_normalization_5/add/ReadVariableOp(layer_normalization_5/add/ReadVariableOp2X
*layer_normalization_5/mul_3/ReadVariableOp*layer_normalization_5/mul_3/ReadVariableOp:\ X
4
_output_shapes"
 :                  `
 
_user_specified_nameinputs
тS
Щ
@__inference_model_layer_call_and_return_conditional_losses_17263
input_1
input_2
input_3
input_4!
node_ide1_17185:@
node_ide1_17187:@'
layer_normalization_17191:@'
layer_normalization_17193:@!
edge_ide1_17196:@
edge_ide1_17198:@!
node_ide2_17201:@`
node_ide2_17203:`)
layer_normalization_1_17208:@)
layer_normalization_1_17210:@!
edge_ide2_17213:@`
edge_ide2_17215:`)
layer_normalization_2_17223:`)
layer_normalization_2_17225:`)
layer_normalization_3_17228:`)
layer_normalization_3_17230:`
	mpn_17236:	а`
	mpn_17238:`
	mpn_17240:`
	mpn_17242:`
	mpn_17244:	└`
	mpn_17246:`
	mpn_17248:`
	mpn_17250:`'
node_prediction_17257:`#
node_prediction_17259:
identityИв!edge_ide1/StatefulPartitionedCallв!edge_ide2/StatefulPartitionedCallв+layer_normalization/StatefulPartitionedCallв-layer_normalization_1/StatefulPartitionedCallв-layer_normalization_2/StatefulPartitionedCallв-layer_normalization_3/StatefulPartitionedCallвmpn/StatefulPartitionedCallв!node_ide1/StatefulPartitionedCallв!node_ide2/StatefulPartitionedCallв'node_prediction/StatefulPartitionedCall 
!node_ide1/StatefulPartitionedCallStatefulPartitionedCallinput_1node_ide1_17185node_ide1_17187*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_node_ide1_layer_call_and_return_conditional_losses_15737ф
lambda/PartitionedCallPartitionedCall*node_ide1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_lambda_layer_call_and_return_conditional_losses_15754┐
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0layer_normalization_17191layer_normalization_17193*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *W
fRRP
N__inference_layer_normalization_layer_call_and_return_conditional_losses_15803 
!edge_ide1/StatefulPartitionedCallStatefulPartitionedCallinput_2edge_ide1_17196edge_ide1_17198*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_edge_ide1_layer_call_and_return_conditional_losses_15839м
!node_ide2/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0node_ide2_17201node_ide2_17203*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  `*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_node_ide2_layer_call_and_return_conditional_losses_15875ц
lambda/PartitionedCall_1PartitionedCall*edge_ide1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_lambda_layer_call_and_return_conditional_losses_15754ц
lambda/PartitionedCall_2PartitionedCall*node_ide2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  `* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_lambda_layer_call_and_return_conditional_losses_15364╔
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCall!lambda/PartitionedCall_1:output:0layer_normalization_1_17208layer_normalization_1_17210*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_layer_normalization_1_layer_call_and_return_conditional_losses_15930о
!edge_ide2/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_1/StatefulPartitionedCall:output:0edge_ide2_17213edge_ide2_17215*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  `*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_edge_ide2_layer_call_and_return_conditional_losses_15966}
,tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
.tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
.tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╔
&tf.__operators__.getitem/strided_sliceStridedSliceinput_25tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*4
_output_shapes"
 :                  *
ellipsis_maskц
lambda/PartitionedCall_3PartitionedCall*edge_ide2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  `* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_lambda_layer_call_and_return_conditional_losses_15364╔
-layer_normalization_2/StatefulPartitionedCallStatefulPartitionedCall!lambda/PartitionedCall_2:output:0layer_normalization_2_17223layer_normalization_2_17225*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  `*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_layer_normalization_2_layer_call_and_return_conditional_losses_16024╔
-layer_normalization_3/StatefulPartitionedCallStatefulPartitionedCall!lambda/PartitionedCall_3:output:0layer_normalization_3_17228layer_normalization_3_17230*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  `*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_layer_normalization_3_layer_call_and_return_conditional_losses_16077{
tf.ones_like/ones_like/ShapeShape/tf.__operators__.getitem/strided_slice:output:0*
T0*
_output_shapes
:a
tf.ones_like/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?л
tf.ones_like/ones_likeFill%tf.ones_like/ones_like/Shape:output:0%tf.ones_like/ones_like/Const:output:0*
T0*4
_output_shapes"
 :                  ┌
mpn/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_2/StatefulPartitionedCall:output:06layer_normalization_3/StatefulPartitionedCall:output:0input_3tf.ones_like/ones_like:output:0input_4	mpn_17236	mpn_17238	mpn_17240	mpn_17242	mpn_17244	mpn_17246	mpn_17248	mpn_17250*
Tin
2*
Tout	
2*
_collective_manager_ids
 *╢
_output_shapesг
а:                  `:                  `:                  :                  :                  **
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8В *G
fBR@
>__inference_mpn_layer_call_and_return_conditional_losses_16354┤
'node_prediction/StatefulPartitionedCallStatefulPartitionedCall$mpn/StatefulPartitionedCall:output:0node_prediction_17257node_prediction_17259*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_node_prediction_layer_call_and_return_conditional_losses_16406М
IdentityIdentity0node_prediction/StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  ▄
NoOpNoOp"^edge_ide1/StatefulPartitionedCall"^edge_ide2/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall.^layer_normalization_1/StatefulPartitionedCall.^layer_normalization_2/StatefulPartitionedCall.^layer_normalization_3/StatefulPartitionedCall^mpn/StatefulPartitionedCall"^node_ide1/StatefulPartitionedCall"^node_ide2/StatefulPartitionedCall(^node_prediction/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*╔
_input_shapes╖
┤:                  :                  :                  :                  : : : : : : : : : : : : : : : : : : : : : : : : : : 2F
!edge_ide1/StatefulPartitionedCall!edge_ide1/StatefulPartitionedCall2F
!edge_ide2/StatefulPartitionedCall!edge_ide2/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2^
-layer_normalization_1/StatefulPartitionedCall-layer_normalization_1/StatefulPartitionedCall2^
-layer_normalization_2/StatefulPartitionedCall-layer_normalization_2/StatefulPartitionedCall2^
-layer_normalization_3/StatefulPartitionedCall-layer_normalization_3/StatefulPartitionedCall2:
mpn/StatefulPartitionedCallmpn/StatefulPartitionedCall2F
!node_ide1/StatefulPartitionedCall!node_ide1/StatefulPartitionedCall2F
!node_ide2/StatefulPartitionedCall!node_ide2/StatefulPartitionedCall2R
'node_prediction/StatefulPartitionedCall'node_prediction/StatefulPartitionedCall:] Y
4
_output_shapes"
 :                  
!
_user_specified_name	input_1:]Y
4
_output_shapes"
 :                  
!
_user_specified_name	input_2:]Y
4
_output_shapes"
 :                  
!
_user_specified_name	input_3:]Y
4
_output_shapes"
 :                  
!
_user_specified_name	input_4
ЦШ
╣
@__inference_model_layer_call_and_return_conditional_losses_18149
inputs_0
inputs_1
inputs_2
inputs_3=
+node_ide1_tensordot_readvariableop_resource:@7
)node_ide1_biasadd_readvariableop_resource:@?
1layer_normalization_mul_3_readvariableop_resource:@=
/layer_normalization_add_readvariableop_resource:@=
+edge_ide1_tensordot_readvariableop_resource:@7
)edge_ide1_biasadd_readvariableop_resource:@=
+node_ide2_tensordot_readvariableop_resource:@`7
)node_ide2_biasadd_readvariableop_resource:`A
3layer_normalization_1_mul_3_readvariableop_resource:@?
1layer_normalization_1_add_readvariableop_resource:@=
+edge_ide2_tensordot_readvariableop_resource:@`7
)edge_ide2_biasadd_readvariableop_resource:`A
3layer_normalization_2_mul_3_readvariableop_resource:`?
1layer_normalization_2_add_readvariableop_resource:`A
3layer_normalization_3_mul_3_readvariableop_resource:`?
1layer_normalization_3_add_readvariableop_resource:`I
6mpn_sequential_dense_tensordot_readvariableop_resource:	а`B
4mpn_sequential_dense_biasadd_readvariableop_resource:`P
Bmpn_sequential_layer_normalization_4_mul_3_readvariableop_resource:`N
@mpn_sequential_layer_normalization_4_add_readvariableop_resource:`@
-mpn_dense_1_tensordot_readvariableop_resource:	└`9
+mpn_dense_1_biasadd_readvariableop_resource:`R
Dmpn_sequential_1_layer_normalization_5_mul_3_readvariableop_resource:`P
Bmpn_sequential_1_layer_normalization_5_add_readvariableop_resource:`C
1node_prediction_tensordot_readvariableop_resource:`=
/node_prediction_biasadd_readvariableop_resource:
identityИв edge_ide1/BiasAdd/ReadVariableOpв"edge_ide1/Tensordot/ReadVariableOpв edge_ide2/BiasAdd/ReadVariableOpв"edge_ide2/Tensordot/ReadVariableOpв&layer_normalization/add/ReadVariableOpв(layer_normalization/mul_3/ReadVariableOpв(layer_normalization_1/add/ReadVariableOpв*layer_normalization_1/mul_3/ReadVariableOpв(layer_normalization_2/add/ReadVariableOpв*layer_normalization_2/mul_3/ReadVariableOpв(layer_normalization_3/add/ReadVariableOpв*layer_normalization_3/mul_3/ReadVariableOpв"mpn/dense_1/BiasAdd/ReadVariableOpв$mpn/dense_1/Tensordot/ReadVariableOpв+mpn/sequential/dense/BiasAdd/ReadVariableOpв-mpn/sequential/dense/Tensordot/ReadVariableOpв7mpn/sequential/layer_normalization_4/add/ReadVariableOpв9mpn/sequential/layer_normalization_4/mul_3/ReadVariableOpв9mpn/sequential_1/layer_normalization_5/add/ReadVariableOpв;mpn/sequential_1/layer_normalization_5/mul_3/ReadVariableOpв node_ide1/BiasAdd/ReadVariableOpв"node_ide1/Tensordot/ReadVariableOpв node_ide2/BiasAdd/ReadVariableOpв"node_ide2/Tensordot/ReadVariableOpв&node_prediction/BiasAdd/ReadVariableOpв(node_prediction/Tensordot/ReadVariableOpО
"node_ide1/Tensordot/ReadVariableOpReadVariableOp+node_ide1_tensordot_readvariableop_resource*
_output_shapes

:@*
dtype0b
node_ide1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:i
node_ide1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       Q
node_ide1/Tensordot/ShapeShapeinputs_0*
T0*
_output_shapes
:c
!node_ide1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : у
node_ide1/Tensordot/GatherV2GatherV2"node_ide1/Tensordot/Shape:output:0!node_ide1/Tensordot/free:output:0*node_ide1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:e
#node_ide1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ч
node_ide1/Tensordot/GatherV2_1GatherV2"node_ide1/Tensordot/Shape:output:0!node_ide1/Tensordot/axes:output:0,node_ide1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
node_ide1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: М
node_ide1/Tensordot/ProdProd%node_ide1/Tensordot/GatherV2:output:0"node_ide1/Tensordot/Const:output:0*
T0*
_output_shapes
: e
node_ide1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Т
node_ide1/Tensordot/Prod_1Prod'node_ide1/Tensordot/GatherV2_1:output:0$node_ide1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: a
node_ide1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ─
node_ide1/Tensordot/concatConcatV2!node_ide1/Tensordot/free:output:0!node_ide1/Tensordot/axes:output:0(node_ide1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Ч
node_ide1/Tensordot/stackPack!node_ide1/Tensordot/Prod:output:0#node_ide1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Ш
node_ide1/Tensordot/transpose	Transposeinputs_0#node_ide1/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :                  и
node_ide1/Tensordot/ReshapeReshape!node_ide1/Tensordot/transpose:y:0"node_ide1/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  и
node_ide1/Tensordot/MatMulMatMul$node_ide1/Tensordot/Reshape:output:0*node_ide1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @e
node_ide1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@c
!node_ide1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ╧
node_ide1/Tensordot/concat_1ConcatV2%node_ide1/Tensordot/GatherV2:output:0$node_ide1/Tensordot/Const_2:output:0*node_ide1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:к
node_ide1/TensordotReshape$node_ide1/Tensordot/MatMul:product:0%node_ide1/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  @Ж
 node_ide1/BiasAdd/ReadVariableOpReadVariableOp)node_ide1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0г
node_ide1/BiasAddBiasAddnode_ide1/Tensordot:output:0(node_ide1/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  @V
lambda/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Н
lambda/Gelu/mulMullambda/Gelu/mul/x:output:0node_ide1/BiasAdd:output:0*
T0*4
_output_shapes"
 :                  @W
lambda/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *є╡?Ц
lambda/Gelu/truedivRealDivnode_ide1/BiasAdd:output:0lambda/Gelu/Cast/x:output:0*
T0*4
_output_shapes"
 :                  @n
lambda/Gelu/ErfErflambda/Gelu/truediv:z:0*
T0*4
_output_shapes"
 :                  @V
lambda/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?И
lambda/Gelu/addAddV2lambda/Gelu/add/x:output:0lambda/Gelu/Erf:y:0*
T0*4
_output_shapes"
 :                  @Б
lambda/Gelu/mul_1Mullambda/Gelu/mul:z:0lambda/Gelu/add:z:0*
T0*4
_output_shapes"
 :                  @^
layer_normalization/ShapeShapelambda/Gelu/mul_1:z:0*
T0*
_output_shapes
:q
'layer_normalization/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)layer_normalization/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)layer_normalization/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╡
!layer_normalization/strided_sliceStridedSlice"layer_normalization/Shape:output:00layer_normalization/strided_slice/stack:output:02layer_normalization/strided_slice/stack_1:output:02layer_normalization/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
layer_normalization/mul/xConst*
_output_shapes
: *
dtype0*
value	B :П
layer_normalization/mulMul"layer_normalization/mul/x:output:0*layer_normalization/strided_slice:output:0*
T0*
_output_shapes
: s
)layer_normalization/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╜
#layer_normalization/strided_slice_1StridedSlice"layer_normalization/Shape:output:02layer_normalization/strided_slice_1/stack:output:04layer_normalization/strided_slice_1/stack_1:output:04layer_normalization/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskМ
layer_normalization/mul_1Mullayer_normalization/mul:z:0,layer_normalization/strided_slice_1:output:0*
T0*
_output_shapes
: s
)layer_normalization/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╜
#layer_normalization/strided_slice_2StridedSlice"layer_normalization/Shape:output:02layer_normalization/strided_slice_2/stack:output:04layer_normalization/strided_slice_2/stack_1:output:04layer_normalization/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :Х
layer_normalization/mul_2Mul$layer_normalization/mul_2/x:output:0,layer_normalization/strided_slice_2:output:0*
T0*
_output_shapes
: e
#layer_normalization/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :e
#layer_normalization/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :ё
!layer_normalization/Reshape/shapePack,layer_normalization/Reshape/shape/0:output:0layer_normalization/mul_1:z:0layer_normalization/mul_2:z:0,layer_normalization/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:г
layer_normalization/ReshapeReshapelambda/Gelu/mul_1:z:0*layer_normalization/Reshape/shape:output:0*
T0*/
_output_shapes
:         @t
layer_normalization/ones/packedPacklayer_normalization/mul_1:z:0*
N*
T0*
_output_shapes
:c
layer_normalization/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?б
layer_normalization/onesFill(layer_normalization/ones/packed:output:0'layer_normalization/ones/Const:output:0*
T0*#
_output_shapes
:         u
 layer_normalization/zeros/packedPacklayer_normalization/mul_1:z:0*
N*
T0*
_output_shapes
:d
layer_normalization/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    д
layer_normalization/zerosFill)layer_normalization/zeros/packed:output:0(layer_normalization/zeros/Const:output:0*
T0*#
_output_shapes
:         \
layer_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB ^
layer_normalization/Const_1Const*
_output_shapes
: *
dtype0*
valueB Ы
$layer_normalization/FusedBatchNormV3FusedBatchNormV3$layer_normalization/Reshape:output:0!layer_normalization/ones:output:0"layer_normalization/zeros:output:0"layer_normalization/Const:output:0$layer_normalization/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:         @:         :         :         :         :*
data_formatNCHW*
epsilon%oГ:╡
layer_normalization/Reshape_1Reshape(layer_normalization/FusedBatchNormV3:y:0"layer_normalization/Shape:output:0*
T0*4
_output_shapes"
 :                  @Ц
(layer_normalization/mul_3/ReadVariableOpReadVariableOp1layer_normalization_mul_3_readvariableop_resource*
_output_shapes
:@*
dtype0╣
layer_normalization/mul_3Mul&layer_normalization/Reshape_1:output:00layer_normalization/mul_3/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  @Т
&layer_normalization/add/ReadVariableOpReadVariableOp/layer_normalization_add_readvariableop_resource*
_output_shapes
:@*
dtype0о
layer_normalization/addAddV2layer_normalization/mul_3:z:0.layer_normalization/add/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  @О
"edge_ide1/Tensordot/ReadVariableOpReadVariableOp+edge_ide1_tensordot_readvariableop_resource*
_output_shapes

:@*
dtype0b
edge_ide1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:i
edge_ide1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       Q
edge_ide1/Tensordot/ShapeShapeinputs_1*
T0*
_output_shapes
:c
!edge_ide1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : у
edge_ide1/Tensordot/GatherV2GatherV2"edge_ide1/Tensordot/Shape:output:0!edge_ide1/Tensordot/free:output:0*edge_ide1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:e
#edge_ide1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ч
edge_ide1/Tensordot/GatherV2_1GatherV2"edge_ide1/Tensordot/Shape:output:0!edge_ide1/Tensordot/axes:output:0,edge_ide1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
edge_ide1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: М
edge_ide1/Tensordot/ProdProd%edge_ide1/Tensordot/GatherV2:output:0"edge_ide1/Tensordot/Const:output:0*
T0*
_output_shapes
: e
edge_ide1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Т
edge_ide1/Tensordot/Prod_1Prod'edge_ide1/Tensordot/GatherV2_1:output:0$edge_ide1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: a
edge_ide1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ─
edge_ide1/Tensordot/concatConcatV2!edge_ide1/Tensordot/free:output:0!edge_ide1/Tensordot/axes:output:0(edge_ide1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Ч
edge_ide1/Tensordot/stackPack!edge_ide1/Tensordot/Prod:output:0#edge_ide1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Ш
edge_ide1/Tensordot/transpose	Transposeinputs_1#edge_ide1/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :                  и
edge_ide1/Tensordot/ReshapeReshape!edge_ide1/Tensordot/transpose:y:0"edge_ide1/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  и
edge_ide1/Tensordot/MatMulMatMul$edge_ide1/Tensordot/Reshape:output:0*edge_ide1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @e
edge_ide1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@c
!edge_ide1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ╧
edge_ide1/Tensordot/concat_1ConcatV2%edge_ide1/Tensordot/GatherV2:output:0$edge_ide1/Tensordot/Const_2:output:0*edge_ide1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:к
edge_ide1/TensordotReshape$edge_ide1/Tensordot/MatMul:product:0%edge_ide1/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  @Ж
 edge_ide1/BiasAdd/ReadVariableOpReadVariableOp)edge_ide1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0г
edge_ide1/BiasAddBiasAddedge_ide1/Tensordot:output:0(edge_ide1/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  @О
"node_ide2/Tensordot/ReadVariableOpReadVariableOp+node_ide2_tensordot_readvariableop_resource*
_output_shapes

:@`*
dtype0b
node_ide2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:i
node_ide2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       d
node_ide2/Tensordot/ShapeShapelayer_normalization/add:z:0*
T0*
_output_shapes
:c
!node_ide2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : у
node_ide2/Tensordot/GatherV2GatherV2"node_ide2/Tensordot/Shape:output:0!node_ide2/Tensordot/free:output:0*node_ide2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:e
#node_ide2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ч
node_ide2/Tensordot/GatherV2_1GatherV2"node_ide2/Tensordot/Shape:output:0!node_ide2/Tensordot/axes:output:0,node_ide2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
node_ide2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: М
node_ide2/Tensordot/ProdProd%node_ide2/Tensordot/GatherV2:output:0"node_ide2/Tensordot/Const:output:0*
T0*
_output_shapes
: e
node_ide2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Т
node_ide2/Tensordot/Prod_1Prod'node_ide2/Tensordot/GatherV2_1:output:0$node_ide2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: a
node_ide2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ─
node_ide2/Tensordot/concatConcatV2!node_ide2/Tensordot/free:output:0!node_ide2/Tensordot/axes:output:0(node_ide2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Ч
node_ide2/Tensordot/stackPack!node_ide2/Tensordot/Prod:output:0#node_ide2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:л
node_ide2/Tensordot/transpose	Transposelayer_normalization/add:z:0#node_ide2/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :                  @и
node_ide2/Tensordot/ReshapeReshape!node_ide2/Tensordot/transpose:y:0"node_ide2/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  и
node_ide2/Tensordot/MatMulMatMul$node_ide2/Tensordot/Reshape:output:0*node_ide2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `e
node_ide2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:`c
!node_ide2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ╧
node_ide2/Tensordot/concat_1ConcatV2%node_ide2/Tensordot/GatherV2:output:0$node_ide2/Tensordot/Const_2:output:0*node_ide2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:к
node_ide2/TensordotReshape$node_ide2/Tensordot/MatMul:product:0%node_ide2/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  `Ж
 node_ide2/BiasAdd/ReadVariableOpReadVariableOp)node_ide2_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0г
node_ide2/BiasAddBiasAddnode_ide2/Tensordot:output:0(node_ide2/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `X
lambda/Gelu_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?С
lambda/Gelu_1/mulMullambda/Gelu_1/mul/x:output:0edge_ide1/BiasAdd:output:0*
T0*4
_output_shapes"
 :                  @Y
lambda/Gelu_1/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *є╡?Ъ
lambda/Gelu_1/truedivRealDivedge_ide1/BiasAdd:output:0lambda/Gelu_1/Cast/x:output:0*
T0*4
_output_shapes"
 :                  @r
lambda/Gelu_1/ErfErflambda/Gelu_1/truediv:z:0*
T0*4
_output_shapes"
 :                  @X
lambda/Gelu_1/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?О
lambda/Gelu_1/addAddV2lambda/Gelu_1/add/x:output:0lambda/Gelu_1/Erf:y:0*
T0*4
_output_shapes"
 :                  @З
lambda/Gelu_1/mul_1Mullambda/Gelu_1/mul:z:0lambda/Gelu_1/add:z:0*
T0*4
_output_shapes"
 :                  @X
lambda/Gelu_2/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?С
lambda/Gelu_2/mulMullambda/Gelu_2/mul/x:output:0node_ide2/BiasAdd:output:0*
T0*4
_output_shapes"
 :                  `Y
lambda/Gelu_2/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *є╡?Ъ
lambda/Gelu_2/truedivRealDivnode_ide2/BiasAdd:output:0lambda/Gelu_2/Cast/x:output:0*
T0*4
_output_shapes"
 :                  `r
lambda/Gelu_2/ErfErflambda/Gelu_2/truediv:z:0*
T0*4
_output_shapes"
 :                  `X
lambda/Gelu_2/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?О
lambda/Gelu_2/addAddV2lambda/Gelu_2/add/x:output:0lambda/Gelu_2/Erf:y:0*
T0*4
_output_shapes"
 :                  `З
lambda/Gelu_2/mul_1Mullambda/Gelu_2/mul:z:0lambda/Gelu_2/add:z:0*
T0*4
_output_shapes"
 :                  `b
layer_normalization_1/ShapeShapelambda/Gelu_1/mul_1:z:0*
T0*
_output_shapes
:s
)layer_normalization_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+layer_normalization_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┐
#layer_normalization_1/strided_sliceStridedSlice$layer_normalization_1/Shape:output:02layer_normalization_1/strided_slice/stack:output:04layer_normalization_1/strided_slice/stack_1:output:04layer_normalization_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization_1/mul/xConst*
_output_shapes
: *
dtype0*
value	B :Х
layer_normalization_1/mulMul$layer_normalization_1/mul/x:output:0,layer_normalization_1/strided_slice:output:0*
T0*
_output_shapes
: u
+layer_normalization_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╟
%layer_normalization_1/strided_slice_1StridedSlice$layer_normalization_1/Shape:output:04layer_normalization_1/strided_slice_1/stack:output:06layer_normalization_1/strided_slice_1/stack_1:output:06layer_normalization_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskТ
layer_normalization_1/mul_1Mullayer_normalization_1/mul:z:0.layer_normalization_1/strided_slice_1:output:0*
T0*
_output_shapes
: u
+layer_normalization_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╟
%layer_normalization_1/strided_slice_2StridedSlice$layer_normalization_1/Shape:output:04layer_normalization_1/strided_slice_2/stack:output:06layer_normalization_1/strided_slice_2/stack_1:output:06layer_normalization_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
layer_normalization_1/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :Ы
layer_normalization_1/mul_2Mul&layer_normalization_1/mul_2/x:output:0.layer_normalization_1/strided_slice_2:output:0*
T0*
_output_shapes
: g
%layer_normalization_1/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :g
%layer_normalization_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :√
#layer_normalization_1/Reshape/shapePack.layer_normalization_1/Reshape/shape/0:output:0layer_normalization_1/mul_1:z:0layer_normalization_1/mul_2:z:0.layer_normalization_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:й
layer_normalization_1/ReshapeReshapelambda/Gelu_1/mul_1:z:0,layer_normalization_1/Reshape/shape:output:0*
T0*/
_output_shapes
:         @x
!layer_normalization_1/ones/packedPacklayer_normalization_1/mul_1:z:0*
N*
T0*
_output_shapes
:e
 layer_normalization_1/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?з
layer_normalization_1/onesFill*layer_normalization_1/ones/packed:output:0)layer_normalization_1/ones/Const:output:0*
T0*#
_output_shapes
:         y
"layer_normalization_1/zeros/packedPacklayer_normalization_1/mul_1:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    к
layer_normalization_1/zerosFill+layer_normalization_1/zeros/packed:output:0*layer_normalization_1/zeros/Const:output:0*
T0*#
_output_shapes
:         ^
layer_normalization_1/ConstConst*
_output_shapes
: *
dtype0*
valueB `
layer_normalization_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB з
&layer_normalization_1/FusedBatchNormV3FusedBatchNormV3&layer_normalization_1/Reshape:output:0#layer_normalization_1/ones:output:0$layer_normalization_1/zeros:output:0$layer_normalization_1/Const:output:0&layer_normalization_1/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:         @:         :         :         :         :*
data_formatNCHW*
epsilon%oГ:╗
layer_normalization_1/Reshape_1Reshape*layer_normalization_1/FusedBatchNormV3:y:0$layer_normalization_1/Shape:output:0*
T0*4
_output_shapes"
 :                  @Ъ
*layer_normalization_1/mul_3/ReadVariableOpReadVariableOp3layer_normalization_1_mul_3_readvariableop_resource*
_output_shapes
:@*
dtype0┐
layer_normalization_1/mul_3Mul(layer_normalization_1/Reshape_1:output:02layer_normalization_1/mul_3/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  @Ц
(layer_normalization_1/add/ReadVariableOpReadVariableOp1layer_normalization_1_add_readvariableop_resource*
_output_shapes
:@*
dtype0┤
layer_normalization_1/addAddV2layer_normalization_1/mul_3:z:00layer_normalization_1/add/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  @О
"edge_ide2/Tensordot/ReadVariableOpReadVariableOp+edge_ide2_tensordot_readvariableop_resource*
_output_shapes

:@`*
dtype0b
edge_ide2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:i
edge_ide2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       f
edge_ide2/Tensordot/ShapeShapelayer_normalization_1/add:z:0*
T0*
_output_shapes
:c
!edge_ide2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : у
edge_ide2/Tensordot/GatherV2GatherV2"edge_ide2/Tensordot/Shape:output:0!edge_ide2/Tensordot/free:output:0*edge_ide2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:e
#edge_ide2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ч
edge_ide2/Tensordot/GatherV2_1GatherV2"edge_ide2/Tensordot/Shape:output:0!edge_ide2/Tensordot/axes:output:0,edge_ide2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
edge_ide2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: М
edge_ide2/Tensordot/ProdProd%edge_ide2/Tensordot/GatherV2:output:0"edge_ide2/Tensordot/Const:output:0*
T0*
_output_shapes
: e
edge_ide2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Т
edge_ide2/Tensordot/Prod_1Prod'edge_ide2/Tensordot/GatherV2_1:output:0$edge_ide2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: a
edge_ide2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ─
edge_ide2/Tensordot/concatConcatV2!edge_ide2/Tensordot/free:output:0!edge_ide2/Tensordot/axes:output:0(edge_ide2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Ч
edge_ide2/Tensordot/stackPack!edge_ide2/Tensordot/Prod:output:0#edge_ide2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:н
edge_ide2/Tensordot/transpose	Transposelayer_normalization_1/add:z:0#edge_ide2/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :                  @и
edge_ide2/Tensordot/ReshapeReshape!edge_ide2/Tensordot/transpose:y:0"edge_ide2/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  и
edge_ide2/Tensordot/MatMulMatMul$edge_ide2/Tensordot/Reshape:output:0*edge_ide2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `e
edge_ide2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:`c
!edge_ide2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ╧
edge_ide2/Tensordot/concat_1ConcatV2%edge_ide2/Tensordot/GatherV2:output:0$edge_ide2/Tensordot/Const_2:output:0*edge_ide2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:к
edge_ide2/TensordotReshape$edge_ide2/Tensordot/MatMul:product:0%edge_ide2/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  `Ж
 edge_ide2/BiasAdd/ReadVariableOpReadVariableOp)edge_ide2_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0г
edge_ide2/BiasAddBiasAddedge_ide2/Tensordot:output:0(edge_ide2/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `}
,tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
.tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
.tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╩
&tf.__operators__.getitem/strided_sliceStridedSliceinputs_15tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*4
_output_shapes"
 :                  *
ellipsis_maskX
lambda/Gelu_3/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?С
lambda/Gelu_3/mulMullambda/Gelu_3/mul/x:output:0edge_ide2/BiasAdd:output:0*
T0*4
_output_shapes"
 :                  `Y
lambda/Gelu_3/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *є╡?Ъ
lambda/Gelu_3/truedivRealDivedge_ide2/BiasAdd:output:0lambda/Gelu_3/Cast/x:output:0*
T0*4
_output_shapes"
 :                  `r
lambda/Gelu_3/ErfErflambda/Gelu_3/truediv:z:0*
T0*4
_output_shapes"
 :                  `X
lambda/Gelu_3/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?О
lambda/Gelu_3/addAddV2lambda/Gelu_3/add/x:output:0lambda/Gelu_3/Erf:y:0*
T0*4
_output_shapes"
 :                  `З
lambda/Gelu_3/mul_1Mullambda/Gelu_3/mul:z:0lambda/Gelu_3/add:z:0*
T0*4
_output_shapes"
 :                  `b
layer_normalization_2/ShapeShapelambda/Gelu_2/mul_1:z:0*
T0*
_output_shapes
:s
)layer_normalization_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+layer_normalization_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┐
#layer_normalization_2/strided_sliceStridedSlice$layer_normalization_2/Shape:output:02layer_normalization_2/strided_slice/stack:output:04layer_normalization_2/strided_slice/stack_1:output:04layer_normalization_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization_2/mul/xConst*
_output_shapes
: *
dtype0*
value	B :Х
layer_normalization_2/mulMul$layer_normalization_2/mul/x:output:0,layer_normalization_2/strided_slice:output:0*
T0*
_output_shapes
: u
+layer_normalization_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╟
%layer_normalization_2/strided_slice_1StridedSlice$layer_normalization_2/Shape:output:04layer_normalization_2/strided_slice_1/stack:output:06layer_normalization_2/strided_slice_1/stack_1:output:06layer_normalization_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskТ
layer_normalization_2/mul_1Mullayer_normalization_2/mul:z:0.layer_normalization_2/strided_slice_1:output:0*
T0*
_output_shapes
: u
+layer_normalization_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╟
%layer_normalization_2/strided_slice_2StridedSlice$layer_normalization_2/Shape:output:04layer_normalization_2/strided_slice_2/stack:output:06layer_normalization_2/strided_slice_2/stack_1:output:06layer_normalization_2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
layer_normalization_2/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :Ы
layer_normalization_2/mul_2Mul&layer_normalization_2/mul_2/x:output:0.layer_normalization_2/strided_slice_2:output:0*
T0*
_output_shapes
: g
%layer_normalization_2/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :g
%layer_normalization_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :√
#layer_normalization_2/Reshape/shapePack.layer_normalization_2/Reshape/shape/0:output:0layer_normalization_2/mul_1:z:0layer_normalization_2/mul_2:z:0.layer_normalization_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:й
layer_normalization_2/ReshapeReshapelambda/Gelu_2/mul_1:z:0,layer_normalization_2/Reshape/shape:output:0*
T0*/
_output_shapes
:         `x
!layer_normalization_2/ones/packedPacklayer_normalization_2/mul_1:z:0*
N*
T0*
_output_shapes
:e
 layer_normalization_2/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?з
layer_normalization_2/onesFill*layer_normalization_2/ones/packed:output:0)layer_normalization_2/ones/Const:output:0*
T0*#
_output_shapes
:         y
"layer_normalization_2/zeros/packedPacklayer_normalization_2/mul_1:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    к
layer_normalization_2/zerosFill+layer_normalization_2/zeros/packed:output:0*layer_normalization_2/zeros/Const:output:0*
T0*#
_output_shapes
:         ^
layer_normalization_2/ConstConst*
_output_shapes
: *
dtype0*
valueB `
layer_normalization_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB з
&layer_normalization_2/FusedBatchNormV3FusedBatchNormV3&layer_normalization_2/Reshape:output:0#layer_normalization_2/ones:output:0$layer_normalization_2/zeros:output:0$layer_normalization_2/Const:output:0&layer_normalization_2/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:         `:         :         :         :         :*
data_formatNCHW*
epsilon%oГ:╗
layer_normalization_2/Reshape_1Reshape*layer_normalization_2/FusedBatchNormV3:y:0$layer_normalization_2/Shape:output:0*
T0*4
_output_shapes"
 :                  `Ъ
*layer_normalization_2/mul_3/ReadVariableOpReadVariableOp3layer_normalization_2_mul_3_readvariableop_resource*
_output_shapes
:`*
dtype0┐
layer_normalization_2/mul_3Mul(layer_normalization_2/Reshape_1:output:02layer_normalization_2/mul_3/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `Ц
(layer_normalization_2/add/ReadVariableOpReadVariableOp1layer_normalization_2_add_readvariableop_resource*
_output_shapes
:`*
dtype0┤
layer_normalization_2/addAddV2layer_normalization_2/mul_3:z:00layer_normalization_2/add/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `b
layer_normalization_3/ShapeShapelambda/Gelu_3/mul_1:z:0*
T0*
_output_shapes
:s
)layer_normalization_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+layer_normalization_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┐
#layer_normalization_3/strided_sliceStridedSlice$layer_normalization_3/Shape:output:02layer_normalization_3/strided_slice/stack:output:04layer_normalization_3/strided_slice/stack_1:output:04layer_normalization_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization_3/mul/xConst*
_output_shapes
: *
dtype0*
value	B :Х
layer_normalization_3/mulMul$layer_normalization_3/mul/x:output:0,layer_normalization_3/strided_slice:output:0*
T0*
_output_shapes
: u
+layer_normalization_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╟
%layer_normalization_3/strided_slice_1StridedSlice$layer_normalization_3/Shape:output:04layer_normalization_3/strided_slice_1/stack:output:06layer_normalization_3/strided_slice_1/stack_1:output:06layer_normalization_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskТ
layer_normalization_3/mul_1Mullayer_normalization_3/mul:z:0.layer_normalization_3/strided_slice_1:output:0*
T0*
_output_shapes
: u
+layer_normalization_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╟
%layer_normalization_3/strided_slice_2StridedSlice$layer_normalization_3/Shape:output:04layer_normalization_3/strided_slice_2/stack:output:06layer_normalization_3/strided_slice_2/stack_1:output:06layer_normalization_3/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
layer_normalization_3/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :Ы
layer_normalization_3/mul_2Mul&layer_normalization_3/mul_2/x:output:0.layer_normalization_3/strided_slice_2:output:0*
T0*
_output_shapes
: g
%layer_normalization_3/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :g
%layer_normalization_3/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :√
#layer_normalization_3/Reshape/shapePack.layer_normalization_3/Reshape/shape/0:output:0layer_normalization_3/mul_1:z:0layer_normalization_3/mul_2:z:0.layer_normalization_3/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:й
layer_normalization_3/ReshapeReshapelambda/Gelu_3/mul_1:z:0,layer_normalization_3/Reshape/shape:output:0*
T0*/
_output_shapes
:         `x
!layer_normalization_3/ones/packedPacklayer_normalization_3/mul_1:z:0*
N*
T0*
_output_shapes
:e
 layer_normalization_3/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?з
layer_normalization_3/onesFill*layer_normalization_3/ones/packed:output:0)layer_normalization_3/ones/Const:output:0*
T0*#
_output_shapes
:         y
"layer_normalization_3/zeros/packedPacklayer_normalization_3/mul_1:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization_3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    к
layer_normalization_3/zerosFill+layer_normalization_3/zeros/packed:output:0*layer_normalization_3/zeros/Const:output:0*
T0*#
_output_shapes
:         ^
layer_normalization_3/ConstConst*
_output_shapes
: *
dtype0*
valueB `
layer_normalization_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB з
&layer_normalization_3/FusedBatchNormV3FusedBatchNormV3&layer_normalization_3/Reshape:output:0#layer_normalization_3/ones:output:0$layer_normalization_3/zeros:output:0$layer_normalization_3/Const:output:0&layer_normalization_3/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:         `:         :         :         :         :*
data_formatNCHW*
epsilon%oГ:╗
layer_normalization_3/Reshape_1Reshape*layer_normalization_3/FusedBatchNormV3:y:0$layer_normalization_3/Shape:output:0*
T0*4
_output_shapes"
 :                  `Ъ
*layer_normalization_3/mul_3/ReadVariableOpReadVariableOp3layer_normalization_3_mul_3_readvariableop_resource*
_output_shapes
:`*
dtype0┐
layer_normalization_3/mul_3Mul(layer_normalization_3/Reshape_1:output:02layer_normalization_3/mul_3/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `Ц
(layer_normalization_3/add/ReadVariableOpReadVariableOp1layer_normalization_3_add_readvariableop_resource*
_output_shapes
:`*
dtype0┤
layer_normalization_3/addAddV2layer_normalization_3/mul_3:z:00layer_normalization_3/add/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `{
tf.ones_like/ones_like/ShapeShape/tf.__operators__.getitem/strided_slice:output:0*
T0*
_output_shapes
:a
tf.ones_like/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?л
tf.ones_like/ones_likeFill%tf.ones_like/ones_like/Shape:output:0%tf.ones_like/ones_like/Const:output:0*
T0*4
_output_shapes"
 :                  V
	mpn/ShapeShapelayer_normalization_2/add:z:0*
T0*
_output_shapes
:a
mpn/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:c
mpn/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
mpn/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
mpn/strided_sliceStridedSlicempn/Shape:output:0 mpn/strided_slice/stack:output:0"mpn/strided_slice/stack_1:output:0"mpn/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskC
mpn/Shape_1Shapeinputs_2*
T0*
_output_shapes
:c
mpn/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:e
mpn/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
mpn/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:я
mpn/strided_slice_1StridedSlicempn/Shape_1:output:0"mpn/strided_slice_1/stack:output:0$mpn/strided_slice_1/stack_1:output:0$mpn/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
mpn/Shape_2Shapelayer_normalization_2/add:z:0*
T0*
_output_shapes
:c
mpn/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: e
mpn/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
mpn/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:я
mpn/strided_slice_2StridedSlicempn/Shape_2:output:0"mpn/strided_slice_2/stack:output:0$mpn/strided_slice_2/stack_1:output:0$mpn/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskS
mpn/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :╒
mpn/GatherV2GatherV2layer_normalization_2/add:z:0inputs_2mpn/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*8
_output_shapes&
$:"                  `*

batch_dimsV
mpn/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :└б
mpn/Reshape/shapePackmpn/strided_slice_2:output:0mpn/strided_slice_1:output:0mpn/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:Й
mpn/ReshapeReshapempn/GatherV2:output:0mpn/Reshape/shape:output:0*
T0*5
_output_shapes#
!:                  └Z
mpn/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
         о

mpn/concatConcatV2mpn/Reshape:output:0layer_normalization_3/add:z:0mpn/concat/axis:output:0*
N*
T0*5
_output_shapes#
!:                  ае
-mpn/sequential/dense/Tensordot/ReadVariableOpReadVariableOp6mpn_sequential_dense_tensordot_readvariableop_resource*
_output_shapes
:	а`*
dtype0m
#mpn/sequential/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:t
#mpn/sequential/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       g
$mpn/sequential/dense/Tensordot/ShapeShapempn/concat:output:0*
T0*
_output_shapes
:n
,mpn/sequential/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : П
'mpn/sequential/dense/Tensordot/GatherV2GatherV2-mpn/sequential/dense/Tensordot/Shape:output:0,mpn/sequential/dense/Tensordot/free:output:05mpn/sequential/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:p
.mpn/sequential/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : У
)mpn/sequential/dense/Tensordot/GatherV2_1GatherV2-mpn/sequential/dense/Tensordot/Shape:output:0,mpn/sequential/dense/Tensordot/axes:output:07mpn/sequential/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
$mpn/sequential/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: н
#mpn/sequential/dense/Tensordot/ProdProd0mpn/sequential/dense/Tensordot/GatherV2:output:0-mpn/sequential/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: p
&mpn/sequential/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: │
%mpn/sequential/dense/Tensordot/Prod_1Prod2mpn/sequential/dense/Tensordot/GatherV2_1:output:0/mpn/sequential/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: l
*mpn/sequential/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ё
%mpn/sequential/dense/Tensordot/concatConcatV2,mpn/sequential/dense/Tensordot/free:output:0,mpn/sequential/dense/Tensordot/axes:output:03mpn/sequential/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:╕
$mpn/sequential/dense/Tensordot/stackPack,mpn/sequential/dense/Tensordot/Prod:output:0.mpn/sequential/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:║
(mpn/sequential/dense/Tensordot/transpose	Transposempn/concat:output:0.mpn/sequential/dense/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:                  а╔
&mpn/sequential/dense/Tensordot/ReshapeReshape,mpn/sequential/dense/Tensordot/transpose:y:0-mpn/sequential/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  ╔
%mpn/sequential/dense/Tensordot/MatMulMatMul/mpn/sequential/dense/Tensordot/Reshape:output:05mpn/sequential/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `p
&mpn/sequential/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:`n
,mpn/sequential/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : √
'mpn/sequential/dense/Tensordot/concat_1ConcatV20mpn/sequential/dense/Tensordot/GatherV2:output:0/mpn/sequential/dense/Tensordot/Const_2:output:05mpn/sequential/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:╦
mpn/sequential/dense/TensordotReshape/mpn/sequential/dense/Tensordot/MatMul:product:00mpn/sequential/dense/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  `Ь
+mpn/sequential/dense/BiasAdd/ReadVariableOpReadVariableOp4mpn_sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0─
mpn/sequential/dense/BiasAddBiasAdd'mpn/sequential/dense/Tensordot:output:03mpn/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `e
 mpn/sequential/lambda/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?╢
mpn/sequential/lambda/Gelu/mulMul)mpn/sequential/lambda/Gelu/mul/x:output:0%mpn/sequential/dense/BiasAdd:output:0*
T0*4
_output_shapes"
 :                  `f
!mpn/sequential/lambda/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *є╡?┐
"mpn/sequential/lambda/Gelu/truedivRealDiv%mpn/sequential/dense/BiasAdd:output:0*mpn/sequential/lambda/Gelu/Cast/x:output:0*
T0*4
_output_shapes"
 :                  `М
mpn/sequential/lambda/Gelu/ErfErf&mpn/sequential/lambda/Gelu/truediv:z:0*
T0*4
_output_shapes"
 :                  `e
 mpn/sequential/lambda/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╡
mpn/sequential/lambda/Gelu/addAddV2)mpn/sequential/lambda/Gelu/add/x:output:0"mpn/sequential/lambda/Gelu/Erf:y:0*
T0*4
_output_shapes"
 :                  `о
 mpn/sequential/lambda/Gelu/mul_1Mul"mpn/sequential/lambda/Gelu/mul:z:0"mpn/sequential/lambda/Gelu/add:z:0*
T0*4
_output_shapes"
 :                  `~
*mpn/sequential/layer_normalization_4/ShapeShape$mpn/sequential/lambda/Gelu/mul_1:z:0*
T0*
_output_shapes
:В
8mpn/sequential/layer_normalization_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Д
:mpn/sequential/layer_normalization_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Д
:mpn/sequential/layer_normalization_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:К
2mpn/sequential/layer_normalization_4/strided_sliceStridedSlice3mpn/sequential/layer_normalization_4/Shape:output:0Ampn/sequential/layer_normalization_4/strided_slice/stack:output:0Cmpn/sequential/layer_normalization_4/strided_slice/stack_1:output:0Cmpn/sequential/layer_normalization_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
*mpn/sequential/layer_normalization_4/mul/xConst*
_output_shapes
: *
dtype0*
value	B :┬
(mpn/sequential/layer_normalization_4/mulMul3mpn/sequential/layer_normalization_4/mul/x:output:0;mpn/sequential/layer_normalization_4/strided_slice:output:0*
T0*
_output_shapes
: Д
:mpn/sequential/layer_normalization_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:Ж
<mpn/sequential/layer_normalization_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ж
<mpn/sequential/layer_normalization_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Т
4mpn/sequential/layer_normalization_4/strided_slice_1StridedSlice3mpn/sequential/layer_normalization_4/Shape:output:0Cmpn/sequential/layer_normalization_4/strided_slice_1/stack:output:0Empn/sequential/layer_normalization_4/strided_slice_1/stack_1:output:0Empn/sequential/layer_normalization_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask┐
*mpn/sequential/layer_normalization_4/mul_1Mul,mpn/sequential/layer_normalization_4/mul:z:0=mpn/sequential/layer_normalization_4/strided_slice_1:output:0*
T0*
_output_shapes
: Д
:mpn/sequential/layer_normalization_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:Ж
<mpn/sequential/layer_normalization_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ж
<mpn/sequential/layer_normalization_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Т
4mpn/sequential/layer_normalization_4/strided_slice_2StridedSlice3mpn/sequential/layer_normalization_4/Shape:output:0Cmpn/sequential/layer_normalization_4/strided_slice_2/stack:output:0Empn/sequential/layer_normalization_4/strided_slice_2/stack_1:output:0Empn/sequential/layer_normalization_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
,mpn/sequential/layer_normalization_4/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :╚
*mpn/sequential/layer_normalization_4/mul_2Mul5mpn/sequential/layer_normalization_4/mul_2/x:output:0=mpn/sequential/layer_normalization_4/strided_slice_2:output:0*
T0*
_output_shapes
: v
4mpn/sequential/layer_normalization_4/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :v
4mpn/sequential/layer_normalization_4/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :╞
2mpn/sequential/layer_normalization_4/Reshape/shapePack=mpn/sequential/layer_normalization_4/Reshape/shape/0:output:0.mpn/sequential/layer_normalization_4/mul_1:z:0.mpn/sequential/layer_normalization_4/mul_2:z:0=mpn/sequential/layer_normalization_4/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:╘
,mpn/sequential/layer_normalization_4/ReshapeReshape$mpn/sequential/lambda/Gelu/mul_1:z:0;mpn/sequential/layer_normalization_4/Reshape/shape:output:0*
T0*/
_output_shapes
:         `Ц
0mpn/sequential/layer_normalization_4/ones/packedPack.mpn/sequential/layer_normalization_4/mul_1:z:0*
N*
T0*
_output_shapes
:t
/mpn/sequential/layer_normalization_4/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╘
)mpn/sequential/layer_normalization_4/onesFill9mpn/sequential/layer_normalization_4/ones/packed:output:08mpn/sequential/layer_normalization_4/ones/Const:output:0*
T0*#
_output_shapes
:         Ч
1mpn/sequential/layer_normalization_4/zeros/packedPack.mpn/sequential/layer_normalization_4/mul_1:z:0*
N*
T0*
_output_shapes
:u
0mpn/sequential/layer_normalization_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ╫
*mpn/sequential/layer_normalization_4/zerosFill:mpn/sequential/layer_normalization_4/zeros/packed:output:09mpn/sequential/layer_normalization_4/zeros/Const:output:0*
T0*#
_output_shapes
:         m
*mpn/sequential/layer_normalization_4/ConstConst*
_output_shapes
: *
dtype0*
valueB o
,mpn/sequential/layer_normalization_4/Const_1Const*
_output_shapes
: *
dtype0*
valueB Б
5mpn/sequential/layer_normalization_4/FusedBatchNormV3FusedBatchNormV35mpn/sequential/layer_normalization_4/Reshape:output:02mpn/sequential/layer_normalization_4/ones:output:03mpn/sequential/layer_normalization_4/zeros:output:03mpn/sequential/layer_normalization_4/Const:output:05mpn/sequential/layer_normalization_4/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:         `:         :         :         :         :*
data_formatNCHW*
epsilon%oГ:ш
.mpn/sequential/layer_normalization_4/Reshape_1Reshape9mpn/sequential/layer_normalization_4/FusedBatchNormV3:y:03mpn/sequential/layer_normalization_4/Shape:output:0*
T0*4
_output_shapes"
 :                  `╕
9mpn/sequential/layer_normalization_4/mul_3/ReadVariableOpReadVariableOpBmpn_sequential_layer_normalization_4_mul_3_readvariableop_resource*
_output_shapes
:`*
dtype0ь
*mpn/sequential/layer_normalization_4/mul_3Mul7mpn/sequential/layer_normalization_4/Reshape_1:output:0Ampn/sequential/layer_normalization_4/mul_3/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `┤
7mpn/sequential/layer_normalization_4/add/ReadVariableOpReadVariableOp@mpn_sequential_layer_normalization_4_add_readvariableop_resource*
_output_shapes
:`*
dtype0с
(mpn/sequential/layer_normalization_4/addAddV2.mpn/sequential/layer_normalization_4/mul_3:z:0?mpn/sequential/layer_normalization_4/add/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `Ь
mpn/mulMul,mpn/sequential/layer_normalization_4/add:z:0tf.ones_like/ones_like:output:0*
T0*4
_output_shapes"
 :                  `T
mpn/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :`
mpn/zeros/packedPackmpn/strided_slice:output:0mpn/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:T
mpn/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    x
	mpn/zerosFillmpn/zeros/packed:output:0mpn/zeros/Const:output:0*
T0*'
_output_shapes
:         `I
mpn/scan/ShapeShapempn/mul:z:0*
T0*
_output_shapes
:f
mpn/scan/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
mpn/scan/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
mpn/scan/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:■
mpn/scan/strided_sliceStridedSlicempn/scan/Shape:output:0%mpn/scan/strided_slice/stack:output:0'mpn/scan/strided_slice/stack_1:output:0'mpn/scan/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
$mpn/scan/TensorArrayV2/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    `   ═
mpn/scan/TensorArrayV2TensorListReserve-mpn/scan/TensorArrayV2/element_shape:output:0mpn/scan/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥w
&mpn/scan/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╤
mpn/scan/TensorArrayV2_1TensorListReserve/mpn/scan/TensorArrayV2_1/element_shape:output:0mpn/scan/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╧w
&mpn/scan/TensorArrayV2_2/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╤
mpn/scan/TensorArrayV2_2TensorListReserve/mpn/scan/TensorArrayV2_2/element_shape:output:0mpn/scan/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥П
>mpn/scan/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    `   Ё
0mpn/scan/TensorArrayUnstack/TensorListFromTensorTensorListFromTensormpn/mul:z:0Gmpn/scan/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥С
@mpn/scan/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ё
2mpn/scan/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensorinputs_2Impn/scan/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╧С
@mpn/scan/TensorArrayUnstack_2/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ё
2mpn/scan/TensorArrayUnstack_2/TensorListFromTensorTensorListFromTensorinputs_3Impn/scan/TensorArrayUnstack_2/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥w
&mpn/scan/TensorArrayV2_3/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    `   ╤
mpn/scan/TensorArrayV2_3TensorListReserve/mpn/scan/TensorArrayV2_3/element_shape:output:0mpn/scan/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥P
mpn/scan/ConstConst*
_output_shapes
: *
dtype0*
value	B : ]
mpn/scan/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : У
mpn/scan/whileStatelessWhile$mpn/scan/while/loop_counter:output:0mpn/scan/strided_slice:output:0mpn/scan/Const:output:0mpn/zeros:output:0!mpn/scan/TensorArrayV2_3:handle:0mpn/scan/strided_slice:output:0@mpn/scan/TensorArrayUnstack/TensorListFromTensor:output_handle:0Bmpn/scan/TensorArrayUnstack_1/TensorListFromTensor:output_handle:0Bmpn/scan/TensorArrayUnstack_2/TensorListFromTensor:output_handle:0mpn/strided_slice:output:0*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : :         `: : : : : : * 
_read_only_resource_inputs
 *
_stateful_parallelism( *%
bodyR
mpn_scan_while_body_17977*%
condR
mpn_scan_while_cond_17976*8
output_shapes'
%: : : :         `: : : : : : К
9mpn/scan/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    `   ц
+mpn/scan/TensorArrayV2Stack/TensorListStackTensorListStackmpn/scan/while:output:4Bmpn/scan/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  `*
element_dtype0c
mpn/lambda_1/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
         р
mpn/lambda_1/concatConcatV2layer_normalization_2/add:z:04mpn/scan/TensorArrayV2Stack/TensorListStack:tensor:0!mpn/lambda_1/concat/axis:output:0*
N*
T0*5
_output_shapes#
!:                  └У
$mpn/dense_1/Tensordot/ReadVariableOpReadVariableOp-mpn_dense_1_tensordot_readvariableop_resource*
_output_shapes
:	└`*
dtype0d
mpn/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:k
mpn/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       g
mpn/dense_1/Tensordot/ShapeShapempn/lambda_1/concat:output:0*
T0*
_output_shapes
:e
#mpn/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ы
mpn/dense_1/Tensordot/GatherV2GatherV2$mpn/dense_1/Tensordot/Shape:output:0#mpn/dense_1/Tensordot/free:output:0,mpn/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:g
%mpn/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : я
 mpn/dense_1/Tensordot/GatherV2_1GatherV2$mpn/dense_1/Tensordot/Shape:output:0#mpn/dense_1/Tensordot/axes:output:0.mpn/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:e
mpn/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: Т
mpn/dense_1/Tensordot/ProdProd'mpn/dense_1/Tensordot/GatherV2:output:0$mpn/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: g
mpn/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Ш
mpn/dense_1/Tensordot/Prod_1Prod)mpn/dense_1/Tensordot/GatherV2_1:output:0&mpn/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: c
!mpn/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ╠
mpn/dense_1/Tensordot/concatConcatV2#mpn/dense_1/Tensordot/free:output:0#mpn/dense_1/Tensordot/axes:output:0*mpn/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Э
mpn/dense_1/Tensordot/stackPack#mpn/dense_1/Tensordot/Prod:output:0%mpn/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:▒
mpn/dense_1/Tensordot/transpose	Transposempn/lambda_1/concat:output:0%mpn/dense_1/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:                  └о
mpn/dense_1/Tensordot/ReshapeReshape#mpn/dense_1/Tensordot/transpose:y:0$mpn/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  о
mpn/dense_1/Tensordot/MatMulMatMul&mpn/dense_1/Tensordot/Reshape:output:0,mpn/dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `g
mpn/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:`e
#mpn/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ╫
mpn/dense_1/Tensordot/concat_1ConcatV2'mpn/dense_1/Tensordot/GatherV2:output:0&mpn/dense_1/Tensordot/Const_2:output:0,mpn/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:░
mpn/dense_1/TensordotReshape&mpn/dense_1/Tensordot/MatMul:product:0'mpn/dense_1/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  `К
"mpn/dense_1/BiasAdd/ReadVariableOpReadVariableOp+mpn_dense_1_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0й
mpn/dense_1/BiasAddBiasAddmpn/dense_1/Tensordot:output:0*mpn/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `g
"mpn/sequential_1/lambda/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?▒
 mpn/sequential_1/lambda/Gelu/mulMul+mpn/sequential_1/lambda/Gelu/mul/x:output:0mpn/dense_1/BiasAdd:output:0*
T0*4
_output_shapes"
 :                  `h
#mpn/sequential_1/lambda/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *є╡?║
$mpn/sequential_1/lambda/Gelu/truedivRealDivmpn/dense_1/BiasAdd:output:0,mpn/sequential_1/lambda/Gelu/Cast/x:output:0*
T0*4
_output_shapes"
 :                  `Р
 mpn/sequential_1/lambda/Gelu/ErfErf(mpn/sequential_1/lambda/Gelu/truediv:z:0*
T0*4
_output_shapes"
 :                  `g
"mpn/sequential_1/lambda/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╗
 mpn/sequential_1/lambda/Gelu/addAddV2+mpn/sequential_1/lambda/Gelu/add/x:output:0$mpn/sequential_1/lambda/Gelu/Erf:y:0*
T0*4
_output_shapes"
 :                  `┤
"mpn/sequential_1/lambda/Gelu/mul_1Mul$mpn/sequential_1/lambda/Gelu/mul:z:0$mpn/sequential_1/lambda/Gelu/add:z:0*
T0*4
_output_shapes"
 :                  `В
,mpn/sequential_1/layer_normalization_5/ShapeShape&mpn/sequential_1/lambda/Gelu/mul_1:z:0*
T0*
_output_shapes
:Д
:mpn/sequential_1/layer_normalization_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Ж
<mpn/sequential_1/layer_normalization_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ж
<mpn/sequential_1/layer_normalization_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ф
4mpn/sequential_1/layer_normalization_5/strided_sliceStridedSlice5mpn/sequential_1/layer_normalization_5/Shape:output:0Cmpn/sequential_1/layer_normalization_5/strided_slice/stack:output:0Empn/sequential_1/layer_normalization_5/strided_slice/stack_1:output:0Empn/sequential_1/layer_normalization_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
,mpn/sequential_1/layer_normalization_5/mul/xConst*
_output_shapes
: *
dtype0*
value	B :╚
*mpn/sequential_1/layer_normalization_5/mulMul5mpn/sequential_1/layer_normalization_5/mul/x:output:0=mpn/sequential_1/layer_normalization_5/strided_slice:output:0*
T0*
_output_shapes
: Ж
<mpn/sequential_1/layer_normalization_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:И
>mpn/sequential_1/layer_normalization_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:И
>mpn/sequential_1/layer_normalization_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ь
6mpn/sequential_1/layer_normalization_5/strided_slice_1StridedSlice5mpn/sequential_1/layer_normalization_5/Shape:output:0Empn/sequential_1/layer_normalization_5/strided_slice_1/stack:output:0Gmpn/sequential_1/layer_normalization_5/strided_slice_1/stack_1:output:0Gmpn/sequential_1/layer_normalization_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask┼
,mpn/sequential_1/layer_normalization_5/mul_1Mul.mpn/sequential_1/layer_normalization_5/mul:z:0?mpn/sequential_1/layer_normalization_5/strided_slice_1:output:0*
T0*
_output_shapes
: Ж
<mpn/sequential_1/layer_normalization_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:И
>mpn/sequential_1/layer_normalization_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:И
>mpn/sequential_1/layer_normalization_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ь
6mpn/sequential_1/layer_normalization_5/strided_slice_2StridedSlice5mpn/sequential_1/layer_normalization_5/Shape:output:0Empn/sequential_1/layer_normalization_5/strided_slice_2/stack:output:0Gmpn/sequential_1/layer_normalization_5/strided_slice_2/stack_1:output:0Gmpn/sequential_1/layer_normalization_5/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskp
.mpn/sequential_1/layer_normalization_5/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :╬
,mpn/sequential_1/layer_normalization_5/mul_2Mul7mpn/sequential_1/layer_normalization_5/mul_2/x:output:0?mpn/sequential_1/layer_normalization_5/strided_slice_2:output:0*
T0*
_output_shapes
: x
6mpn/sequential_1/layer_normalization_5/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :x
6mpn/sequential_1/layer_normalization_5/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :╨
4mpn/sequential_1/layer_normalization_5/Reshape/shapePack?mpn/sequential_1/layer_normalization_5/Reshape/shape/0:output:00mpn/sequential_1/layer_normalization_5/mul_1:z:00mpn/sequential_1/layer_normalization_5/mul_2:z:0?mpn/sequential_1/layer_normalization_5/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:┌
.mpn/sequential_1/layer_normalization_5/ReshapeReshape&mpn/sequential_1/lambda/Gelu/mul_1:z:0=mpn/sequential_1/layer_normalization_5/Reshape/shape:output:0*
T0*/
_output_shapes
:         `Ъ
2mpn/sequential_1/layer_normalization_5/ones/packedPack0mpn/sequential_1/layer_normalization_5/mul_1:z:0*
N*
T0*
_output_shapes
:v
1mpn/sequential_1/layer_normalization_5/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?┌
+mpn/sequential_1/layer_normalization_5/onesFill;mpn/sequential_1/layer_normalization_5/ones/packed:output:0:mpn/sequential_1/layer_normalization_5/ones/Const:output:0*
T0*#
_output_shapes
:         Ы
3mpn/sequential_1/layer_normalization_5/zeros/packedPack0mpn/sequential_1/layer_normalization_5/mul_1:z:0*
N*
T0*
_output_shapes
:w
2mpn/sequential_1/layer_normalization_5/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ▌
,mpn/sequential_1/layer_normalization_5/zerosFill<mpn/sequential_1/layer_normalization_5/zeros/packed:output:0;mpn/sequential_1/layer_normalization_5/zeros/Const:output:0*
T0*#
_output_shapes
:         o
,mpn/sequential_1/layer_normalization_5/ConstConst*
_output_shapes
: *
dtype0*
valueB q
.mpn/sequential_1/layer_normalization_5/Const_1Const*
_output_shapes
: *
dtype0*
valueB Н
7mpn/sequential_1/layer_normalization_5/FusedBatchNormV3FusedBatchNormV37mpn/sequential_1/layer_normalization_5/Reshape:output:04mpn/sequential_1/layer_normalization_5/ones:output:05mpn/sequential_1/layer_normalization_5/zeros:output:05mpn/sequential_1/layer_normalization_5/Const:output:07mpn/sequential_1/layer_normalization_5/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:         `:         :         :         :         :*
data_formatNCHW*
epsilon%oГ:ю
0mpn/sequential_1/layer_normalization_5/Reshape_1Reshape;mpn/sequential_1/layer_normalization_5/FusedBatchNormV3:y:05mpn/sequential_1/layer_normalization_5/Shape:output:0*
T0*4
_output_shapes"
 :                  `╝
;mpn/sequential_1/layer_normalization_5/mul_3/ReadVariableOpReadVariableOpDmpn_sequential_1_layer_normalization_5_mul_3_readvariableop_resource*
_output_shapes
:`*
dtype0Є
,mpn/sequential_1/layer_normalization_5/mul_3Mul9mpn/sequential_1/layer_normalization_5/Reshape_1:output:0Cmpn/sequential_1/layer_normalization_5/mul_3/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `╕
9mpn/sequential_1/layer_normalization_5/add/ReadVariableOpReadVariableOpBmpn_sequential_1_layer_normalization_5_add_readvariableop_resource*
_output_shapes
:`*
dtype0ч
*mpn/sequential_1/layer_normalization_5/addAddV20mpn/sequential_1/layer_normalization_5/mul_3:z:0Ampn/sequential_1/layer_normalization_5/add/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `Ъ
(node_prediction/Tensordot/ReadVariableOpReadVariableOp1node_prediction_tensordot_readvariableop_resource*
_output_shapes

:`*
dtype0h
node_prediction/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:o
node_prediction/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       }
node_prediction/Tensordot/ShapeShape.mpn/sequential_1/layer_normalization_5/add:z:0*
T0*
_output_shapes
:i
'node_prediction/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : √
"node_prediction/Tensordot/GatherV2GatherV2(node_prediction/Tensordot/Shape:output:0'node_prediction/Tensordot/free:output:00node_prediction/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:k
)node_prediction/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B :  
$node_prediction/Tensordot/GatherV2_1GatherV2(node_prediction/Tensordot/Shape:output:0'node_prediction/Tensordot/axes:output:02node_prediction/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:i
node_prediction/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: Ю
node_prediction/Tensordot/ProdProd+node_prediction/Tensordot/GatherV2:output:0(node_prediction/Tensordot/Const:output:0*
T0*
_output_shapes
: k
!node_prediction/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: д
 node_prediction/Tensordot/Prod_1Prod-node_prediction/Tensordot/GatherV2_1:output:0*node_prediction/Tensordot/Const_1:output:0*
T0*
_output_shapes
: g
%node_prediction/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ▄
 node_prediction/Tensordot/concatConcatV2'node_prediction/Tensordot/free:output:0'node_prediction/Tensordot/axes:output:0.node_prediction/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:й
node_prediction/Tensordot/stackPack'node_prediction/Tensordot/Prod:output:0)node_prediction/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:╩
#node_prediction/Tensordot/transpose	Transpose.mpn/sequential_1/layer_normalization_5/add:z:0)node_prediction/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :                  `║
!node_prediction/Tensordot/ReshapeReshape'node_prediction/Tensordot/transpose:y:0(node_prediction/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  ║
 node_prediction/Tensordot/MatMulMatMul*node_prediction/Tensordot/Reshape:output:00node_prediction/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         k
!node_prediction/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:i
'node_prediction/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ч
"node_prediction/Tensordot/concat_1ConcatV2+node_prediction/Tensordot/GatherV2:output:0*node_prediction/Tensordot/Const_2:output:00node_prediction/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:╝
node_prediction/TensordotReshape*node_prediction/Tensordot/MatMul:product:0+node_prediction/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  Т
&node_prediction/BiasAdd/ReadVariableOpReadVariableOp/node_prediction_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0╡
node_prediction/BiasAddBiasAdd"node_prediction/Tensordot:output:0.node_prediction/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  |
IdentityIdentity node_prediction/BiasAdd:output:0^NoOp*
T0*4
_output_shapes"
 :                  ░	
NoOpNoOp!^edge_ide1/BiasAdd/ReadVariableOp#^edge_ide1/Tensordot/ReadVariableOp!^edge_ide2/BiasAdd/ReadVariableOp#^edge_ide2/Tensordot/ReadVariableOp'^layer_normalization/add/ReadVariableOp)^layer_normalization/mul_3/ReadVariableOp)^layer_normalization_1/add/ReadVariableOp+^layer_normalization_1/mul_3/ReadVariableOp)^layer_normalization_2/add/ReadVariableOp+^layer_normalization_2/mul_3/ReadVariableOp)^layer_normalization_3/add/ReadVariableOp+^layer_normalization_3/mul_3/ReadVariableOp#^mpn/dense_1/BiasAdd/ReadVariableOp%^mpn/dense_1/Tensordot/ReadVariableOp,^mpn/sequential/dense/BiasAdd/ReadVariableOp.^mpn/sequential/dense/Tensordot/ReadVariableOp8^mpn/sequential/layer_normalization_4/add/ReadVariableOp:^mpn/sequential/layer_normalization_4/mul_3/ReadVariableOp:^mpn/sequential_1/layer_normalization_5/add/ReadVariableOp<^mpn/sequential_1/layer_normalization_5/mul_3/ReadVariableOp!^node_ide1/BiasAdd/ReadVariableOp#^node_ide1/Tensordot/ReadVariableOp!^node_ide2/BiasAdd/ReadVariableOp#^node_ide2/Tensordot/ReadVariableOp'^node_prediction/BiasAdd/ReadVariableOp)^node_prediction/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*╔
_input_shapes╖
┤:                  :                  :                  :                  : : : : : : : : : : : : : : : : : : : : : : : : : : 2D
 edge_ide1/BiasAdd/ReadVariableOp edge_ide1/BiasAdd/ReadVariableOp2H
"edge_ide1/Tensordot/ReadVariableOp"edge_ide1/Tensordot/ReadVariableOp2D
 edge_ide2/BiasAdd/ReadVariableOp edge_ide2/BiasAdd/ReadVariableOp2H
"edge_ide2/Tensordot/ReadVariableOp"edge_ide2/Tensordot/ReadVariableOp2P
&layer_normalization/add/ReadVariableOp&layer_normalization/add/ReadVariableOp2T
(layer_normalization/mul_3/ReadVariableOp(layer_normalization/mul_3/ReadVariableOp2T
(layer_normalization_1/add/ReadVariableOp(layer_normalization_1/add/ReadVariableOp2X
*layer_normalization_1/mul_3/ReadVariableOp*layer_normalization_1/mul_3/ReadVariableOp2T
(layer_normalization_2/add/ReadVariableOp(layer_normalization_2/add/ReadVariableOp2X
*layer_normalization_2/mul_3/ReadVariableOp*layer_normalization_2/mul_3/ReadVariableOp2T
(layer_normalization_3/add/ReadVariableOp(layer_normalization_3/add/ReadVariableOp2X
*layer_normalization_3/mul_3/ReadVariableOp*layer_normalization_3/mul_3/ReadVariableOp2H
"mpn/dense_1/BiasAdd/ReadVariableOp"mpn/dense_1/BiasAdd/ReadVariableOp2L
$mpn/dense_1/Tensordot/ReadVariableOp$mpn/dense_1/Tensordot/ReadVariableOp2Z
+mpn/sequential/dense/BiasAdd/ReadVariableOp+mpn/sequential/dense/BiasAdd/ReadVariableOp2^
-mpn/sequential/dense/Tensordot/ReadVariableOp-mpn/sequential/dense/Tensordot/ReadVariableOp2r
7mpn/sequential/layer_normalization_4/add/ReadVariableOp7mpn/sequential/layer_normalization_4/add/ReadVariableOp2v
9mpn/sequential/layer_normalization_4/mul_3/ReadVariableOp9mpn/sequential/layer_normalization_4/mul_3/ReadVariableOp2v
9mpn/sequential_1/layer_normalization_5/add/ReadVariableOp9mpn/sequential_1/layer_normalization_5/add/ReadVariableOp2z
;mpn/sequential_1/layer_normalization_5/mul_3/ReadVariableOp;mpn/sequential_1/layer_normalization_5/mul_3/ReadVariableOp2D
 node_ide1/BiasAdd/ReadVariableOp node_ide1/BiasAdd/ReadVariableOp2H
"node_ide1/Tensordot/ReadVariableOp"node_ide1/Tensordot/ReadVariableOp2D
 node_ide2/BiasAdd/ReadVariableOp node_ide2/BiasAdd/ReadVariableOp2H
"node_ide2/Tensordot/ReadVariableOp"node_ide2/Tensordot/ReadVariableOp2P
&node_prediction/BiasAdd/ReadVariableOp&node_prediction/BiasAdd/ReadVariableOp2T
(node_prediction/Tensordot/ReadVariableOp(node_prediction/Tensordot/ReadVariableOp:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_0:^Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_1:^Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_2:^Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_3
╝	
]
A__inference_lambda_layer_call_and_return_conditional_losses_18874

inputs
identityO

Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?k
Gelu/mulMulGelu/mul/x:output:0inputs*
T0*4
_output_shapes"
 :                  @P
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *є╡?t
Gelu/truedivRealDivinputsGelu/Cast/x:output:0*
T0*4
_output_shapes"
 :                  @`
Gelu/ErfErfGelu/truediv:z:0*
T0*4
_output_shapes"
 :                  @O

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?s
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*4
_output_shapes"
 :                  @l

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*4
_output_shapes"
 :                  @c
IdentityIdentityGelu/mul_1:z:0*
T0*4
_output_shapes"
 :                  @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :                  @:\ X
4
_output_shapes"
 :                  @
 
_user_specified_nameinputs
║
┐
%__inference_model_layer_call_fn_17471
inputs_0
inputs_1
inputs_2
inputs_3
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@`
	unknown_6:`
	unknown_7:@
	unknown_8:@
	unknown_9:@`

unknown_10:`

unknown_11:`

unknown_12:`

unknown_13:`

unknown_14:`

unknown_15:	а`

unknown_16:`

unknown_17:`

unknown_18:`

unknown_19:	└`

unknown_20:`

unknown_21:`

unknown_22:`

unknown_23:`

unknown_24:
identityИвStatefulPartitionedCall╠
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2inputs_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  *<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_16413|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*╔
_input_shapes╖
┤:                  :                  :                  :                  : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_0:^Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_1:^Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_2:^Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_3
ЇШ
Х&
__inference__traced_save_20629
file_prefix/
+savev2_node_ide1_kernel_read_readvariableop-
)savev2_node_ide1_bias_read_readvariableop8
4savev2_layer_normalization_gamma_read_readvariableop7
3savev2_layer_normalization_beta_read_readvariableop/
+savev2_node_ide2_kernel_read_readvariableop-
)savev2_node_ide2_bias_read_readvariableop/
+savev2_edge_ide1_kernel_read_readvariableop-
)savev2_edge_ide1_bias_read_readvariableop:
6savev2_layer_normalization_1_gamma_read_readvariableop9
5savev2_layer_normalization_1_beta_read_readvariableop/
+savev2_edge_ide2_kernel_read_readvariableop-
)savev2_edge_ide2_bias_read_readvariableop:
6savev2_layer_normalization_2_gamma_read_readvariableop9
5savev2_layer_normalization_2_beta_read_readvariableop:
6savev2_layer_normalization_3_gamma_read_readvariableop9
5savev2_layer_normalization_3_beta_read_readvariableop5
1savev2_node_prediction_kernel_read_readvariableop3
/savev2_node_prediction_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop:
6savev2_layer_normalization_4_gamma_read_readvariableop9
5savev2_layer_normalization_4_beta_read_readvariableop1
-savev2_mpn_dense_1_kernel_read_readvariableop/
+savev2_mpn_dense_1_bias_read_readvariableop:
6savev2_layer_normalization_5_gamma_read_readvariableop9
5savev2_layer_normalization_5_beta_read_readvariableop(
$savev2_iteration_read_readvariableop	,
(savev2_learning_rate_read_readvariableop6
2savev2_adam_m_node_ide1_kernel_read_readvariableop6
2savev2_adam_v_node_ide1_kernel_read_readvariableop4
0savev2_adam_m_node_ide1_bias_read_readvariableop4
0savev2_adam_v_node_ide1_bias_read_readvariableop?
;savev2_adam_m_layer_normalization_gamma_read_readvariableop?
;savev2_adam_v_layer_normalization_gamma_read_readvariableop>
:savev2_adam_m_layer_normalization_beta_read_readvariableop>
:savev2_adam_v_layer_normalization_beta_read_readvariableop6
2savev2_adam_m_node_ide2_kernel_read_readvariableop6
2savev2_adam_v_node_ide2_kernel_read_readvariableop4
0savev2_adam_m_node_ide2_bias_read_readvariableop4
0savev2_adam_v_node_ide2_bias_read_readvariableop6
2savev2_adam_m_edge_ide1_kernel_read_readvariableop6
2savev2_adam_v_edge_ide1_kernel_read_readvariableop4
0savev2_adam_m_edge_ide1_bias_read_readvariableop4
0savev2_adam_v_edge_ide1_bias_read_readvariableopA
=savev2_adam_m_layer_normalization_1_gamma_read_readvariableopA
=savev2_adam_v_layer_normalization_1_gamma_read_readvariableop@
<savev2_adam_m_layer_normalization_1_beta_read_readvariableop@
<savev2_adam_v_layer_normalization_1_beta_read_readvariableop6
2savev2_adam_m_edge_ide2_kernel_read_readvariableop6
2savev2_adam_v_edge_ide2_kernel_read_readvariableop4
0savev2_adam_m_edge_ide2_bias_read_readvariableop4
0savev2_adam_v_edge_ide2_bias_read_readvariableopA
=savev2_adam_m_layer_normalization_2_gamma_read_readvariableopA
=savev2_adam_v_layer_normalization_2_gamma_read_readvariableop@
<savev2_adam_m_layer_normalization_2_beta_read_readvariableop@
<savev2_adam_v_layer_normalization_2_beta_read_readvariableopA
=savev2_adam_m_layer_normalization_3_gamma_read_readvariableopA
=savev2_adam_v_layer_normalization_3_gamma_read_readvariableop@
<savev2_adam_m_layer_normalization_3_beta_read_readvariableop@
<savev2_adam_v_layer_normalization_3_beta_read_readvariableop2
.savev2_adam_m_dense_kernel_read_readvariableop2
.savev2_adam_v_dense_kernel_read_readvariableop0
,savev2_adam_m_dense_bias_read_readvariableop0
,savev2_adam_v_dense_bias_read_readvariableopA
=savev2_adam_m_layer_normalization_4_gamma_read_readvariableopA
=savev2_adam_v_layer_normalization_4_gamma_read_readvariableop@
<savev2_adam_m_layer_normalization_4_beta_read_readvariableop@
<savev2_adam_v_layer_normalization_4_beta_read_readvariableop8
4savev2_adam_m_mpn_dense_1_kernel_read_readvariableop8
4savev2_adam_v_mpn_dense_1_kernel_read_readvariableop6
2savev2_adam_m_mpn_dense_1_bias_read_readvariableop6
2savev2_adam_v_mpn_dense_1_bias_read_readvariableopA
=savev2_adam_m_layer_normalization_5_gamma_read_readvariableopA
=savev2_adam_v_layer_normalization_5_gamma_read_readvariableop@
<savev2_adam_m_layer_normalization_5_beta_read_readvariableop@
<savev2_adam_v_layer_normalization_5_beta_read_readvariableop<
8savev2_adam_m_node_prediction_kernel_read_readvariableop<
8savev2_adam_v_node_prediction_kernel_read_readvariableop:
6savev2_adam_m_node_prediction_bias_read_readvariableop:
6savev2_adam_v_node_prediction_bias_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop
savev2_const

identity_1ИвMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/partБ
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Є"
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:U*
dtype0*Ы"
valueС"BО"UB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/49/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/50/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/51/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/52/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЪ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:U*
dtype0*┐
value╡B▓UB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ж%
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_node_ide1_kernel_read_readvariableop)savev2_node_ide1_bias_read_readvariableop4savev2_layer_normalization_gamma_read_readvariableop3savev2_layer_normalization_beta_read_readvariableop+savev2_node_ide2_kernel_read_readvariableop)savev2_node_ide2_bias_read_readvariableop+savev2_edge_ide1_kernel_read_readvariableop)savev2_edge_ide1_bias_read_readvariableop6savev2_layer_normalization_1_gamma_read_readvariableop5savev2_layer_normalization_1_beta_read_readvariableop+savev2_edge_ide2_kernel_read_readvariableop)savev2_edge_ide2_bias_read_readvariableop6savev2_layer_normalization_2_gamma_read_readvariableop5savev2_layer_normalization_2_beta_read_readvariableop6savev2_layer_normalization_3_gamma_read_readvariableop5savev2_layer_normalization_3_beta_read_readvariableop1savev2_node_prediction_kernel_read_readvariableop/savev2_node_prediction_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop6savev2_layer_normalization_4_gamma_read_readvariableop5savev2_layer_normalization_4_beta_read_readvariableop-savev2_mpn_dense_1_kernel_read_readvariableop+savev2_mpn_dense_1_bias_read_readvariableop6savev2_layer_normalization_5_gamma_read_readvariableop5savev2_layer_normalization_5_beta_read_readvariableop$savev2_iteration_read_readvariableop(savev2_learning_rate_read_readvariableop2savev2_adam_m_node_ide1_kernel_read_readvariableop2savev2_adam_v_node_ide1_kernel_read_readvariableop0savev2_adam_m_node_ide1_bias_read_readvariableop0savev2_adam_v_node_ide1_bias_read_readvariableop;savev2_adam_m_layer_normalization_gamma_read_readvariableop;savev2_adam_v_layer_normalization_gamma_read_readvariableop:savev2_adam_m_layer_normalization_beta_read_readvariableop:savev2_adam_v_layer_normalization_beta_read_readvariableop2savev2_adam_m_node_ide2_kernel_read_readvariableop2savev2_adam_v_node_ide2_kernel_read_readvariableop0savev2_adam_m_node_ide2_bias_read_readvariableop0savev2_adam_v_node_ide2_bias_read_readvariableop2savev2_adam_m_edge_ide1_kernel_read_readvariableop2savev2_adam_v_edge_ide1_kernel_read_readvariableop0savev2_adam_m_edge_ide1_bias_read_readvariableop0savev2_adam_v_edge_ide1_bias_read_readvariableop=savev2_adam_m_layer_normalization_1_gamma_read_readvariableop=savev2_adam_v_layer_normalization_1_gamma_read_readvariableop<savev2_adam_m_layer_normalization_1_beta_read_readvariableop<savev2_adam_v_layer_normalization_1_beta_read_readvariableop2savev2_adam_m_edge_ide2_kernel_read_readvariableop2savev2_adam_v_edge_ide2_kernel_read_readvariableop0savev2_adam_m_edge_ide2_bias_read_readvariableop0savev2_adam_v_edge_ide2_bias_read_readvariableop=savev2_adam_m_layer_normalization_2_gamma_read_readvariableop=savev2_adam_v_layer_normalization_2_gamma_read_readvariableop<savev2_adam_m_layer_normalization_2_beta_read_readvariableop<savev2_adam_v_layer_normalization_2_beta_read_readvariableop=savev2_adam_m_layer_normalization_3_gamma_read_readvariableop=savev2_adam_v_layer_normalization_3_gamma_read_readvariableop<savev2_adam_m_layer_normalization_3_beta_read_readvariableop<savev2_adam_v_layer_normalization_3_beta_read_readvariableop.savev2_adam_m_dense_kernel_read_readvariableop.savev2_adam_v_dense_kernel_read_readvariableop,savev2_adam_m_dense_bias_read_readvariableop,savev2_adam_v_dense_bias_read_readvariableop=savev2_adam_m_layer_normalization_4_gamma_read_readvariableop=savev2_adam_v_layer_normalization_4_gamma_read_readvariableop<savev2_adam_m_layer_normalization_4_beta_read_readvariableop<savev2_adam_v_layer_normalization_4_beta_read_readvariableop4savev2_adam_m_mpn_dense_1_kernel_read_readvariableop4savev2_adam_v_mpn_dense_1_kernel_read_readvariableop2savev2_adam_m_mpn_dense_1_bias_read_readvariableop2savev2_adam_v_mpn_dense_1_bias_read_readvariableop=savev2_adam_m_layer_normalization_5_gamma_read_readvariableop=savev2_adam_v_layer_normalization_5_gamma_read_readvariableop<savev2_adam_m_layer_normalization_5_beta_read_readvariableop<savev2_adam_v_layer_normalization_5_beta_read_readvariableop8savev2_adam_m_node_prediction_kernel_read_readvariableop8savev2_adam_v_node_prediction_kernel_read_readvariableop6savev2_adam_m_node_prediction_bias_read_readvariableop6savev2_adam_v_node_prediction_bias_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *c
dtypesY
W2U	Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:│
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*╙
_input_shapes┴
╛: :@:@:@:@:@`:`:@:@:@:@:@`:`:`:`:`:`:`::	а`:`:`:`:	└`:`:`:`: : :@:@:@:@:@:@:@:@:@`:@`:`:`:@:@:@:@:@:@:@:@:@`:@`:`:`:`:`:`:`:`:`:`:`:	а`:	а`:`:`:`:`:`:`:	└`:	└`:`:`:`:`:`:`:`:`::: : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:$ 

_output_shapes

:@`: 

_output_shapes
:`:$ 

_output_shapes

:@: 

_output_shapes
:@: 	

_output_shapes
:@: 


_output_shapes
:@:$ 

_output_shapes

:@`: 

_output_shapes
:`: 

_output_shapes
:`: 

_output_shapes
:`: 

_output_shapes
:`: 

_output_shapes
:`:$ 

_output_shapes

:`: 

_output_shapes
::%!

_output_shapes
:	а`: 

_output_shapes
:`: 

_output_shapes
:`: 

_output_shapes
:`:%!

_output_shapes
:	└`: 

_output_shapes
:`: 

_output_shapes
:`: 

_output_shapes
:`:

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:@:$ 

_output_shapes

:@: 

_output_shapes
:@:  

_output_shapes
:@: !

_output_shapes
:@: "

_output_shapes
:@: #

_output_shapes
:@: $

_output_shapes
:@:$% 

_output_shapes

:@`:$& 

_output_shapes

:@`: '

_output_shapes
:`: (

_output_shapes
:`:$) 

_output_shapes

:@:$* 

_output_shapes

:@: +

_output_shapes
:@: ,

_output_shapes
:@: -

_output_shapes
:@: .

_output_shapes
:@: /

_output_shapes
:@: 0

_output_shapes
:@:$1 

_output_shapes

:@`:$2 

_output_shapes

:@`: 3

_output_shapes
:`: 4

_output_shapes
:`: 5

_output_shapes
:`: 6

_output_shapes
:`: 7

_output_shapes
:`: 8

_output_shapes
:`: 9

_output_shapes
:`: :

_output_shapes
:`: ;

_output_shapes
:`: <

_output_shapes
:`:%=!

_output_shapes
:	а`:%>!

_output_shapes
:	а`: ?

_output_shapes
:`: @

_output_shapes
:`: A

_output_shapes
:`: B

_output_shapes
:`: C

_output_shapes
:`: D

_output_shapes
:`:%E!

_output_shapes
:	└`:%F!

_output_shapes
:	└`: G

_output_shapes
:`: H

_output_shapes
:`: I

_output_shapes
:`: J

_output_shapes
:`: K

_output_shapes
:`: L

_output_shapes
:`:$M 

_output_shapes

:`:$N 

_output_shapes

:`: O

_output_shapes
:: P

_output_shapes
::Q

_output_shapes
: :R

_output_shapes
: :S

_output_shapes
: :T

_output_shapes
: :U

_output_shapes
: 
Ў
Ц
)__inference_node_ide1_layer_call_fn_18776

inputs
unknown:@
	unknown_0:@
identityИвStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_node_ide1_layer_call_and_return_conditional_losses_15737|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:                  : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
Р%
є
P__inference_layer_normalization_1_layer_call_and_return_conditional_losses_15930

inputs+
mul_3_readvariableop_resource:@)
add_readvariableop_resource:@
identityИвadd/ReadVariableOpвmul_3/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/xConst*
_output_shapes
: *
dtype0*
value	B :S
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: _
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
mul_1Mulmul:z:0strided_slice_1:output:0*
T0*
_output_shapes
: _
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_2Mulmul_2/x:output:0strided_slice_2:output:0*
T0*
_output_shapes
: Q
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Н
Reshape/shapePackReshape/shape/0:output:0	mul_1:z:0	mul_2:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:         @L
ones/packedPack	mul_1:z:0*
N*
T0*
_output_shapes
:O

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?e
onesFillones/packed:output:0ones/Const:output:0*
T0*#
_output_shapes
:         M
zeros/packedPack	mul_1:z:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    h
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*#
_output_shapes
:         H
ConstConst*
_output_shapes
: *
dtype0*
valueB J
Const_1Const*
_output_shapes
: *
dtype0*
valueB г
FusedBatchNormV3FusedBatchNormV3Reshape:output:0ones:output:0zeros:output:0Const:output:0Const_1:output:0*
T0*
U0*o
_output_shapes]
[:         @:         :         :         :         :*
data_formatNCHW*
epsilon%oГ:y
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*4
_output_shapes"
 :                  @n
mul_3/ReadVariableOpReadVariableOpmul_3_readvariableop_resource*
_output_shapes
:@*
dtype0}
mul_3MulReshape_1:output:0mul_3/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  @j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:@*
dtype0r
addAddV2	mul_3:z:0add/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  @c
IdentityIdentityadd:z:0^NoOp*
T0*4
_output_shapes"
 :                  @r
NoOpNoOp^add/ReadVariableOp^mul_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:                  @: : 2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_3/ReadVariableOpmul_3/ReadVariableOp:\ X
4
_output_shapes"
 :                  @
 
_user_specified_nameinputs
К
Ю
5__inference_layer_normalization_5_layer_call_fn_20304

inputs
unknown:`
	unknown_0:`
identityИвStatefulPartitionedCallЄ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  `*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_layer_normalization_5_layer_call_and_return_conditional_losses_15613|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  ``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:                  `: : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :                  `
 
_user_specified_nameinputs
Ы
ў
scan_while_cond_16642&
"scan_while_scan_while_loop_counter!
scan_while_scan_strided_slice
scan_while_placeholder
scan_while_placeholder_1
scan_while_placeholder_2&
"scan_while_less_scan_strided_slice=
9scan_while_scan_while_cond_16642___redundant_placeholder0=
9scan_while_scan_while_cond_16642___redundant_placeholder1=
9scan_while_scan_while_cond_16642___redundant_placeholder2=
9scan_while_scan_while_cond_16642___redundant_placeholder3
scan_while_identity
t
scan/while/LessLessscan_while_placeholder"scan_while_less_scan_strided_slice*
T0*
_output_shapes
: }
scan/while/Less_1Less"scan_while_scan_while_loop_counterscan_while_scan_strided_slice*
T0*
_output_shapes
: g
scan/while/LogicalAnd
LogicalAndscan/while/Less_1:z:0scan/while/Less:z:0*
_output_shapes
: [
scan/while/IdentityIdentityscan/while/LogicalAnd:z:0*
T0
*
_output_shapes
: "3
scan_while_identityscan/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : :         `: : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         `:

_output_shapes
: :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
:
Г
√
D__inference_edge_ide1_layer_call_and_return_conditional_losses_15839

inputs3
!tensordot_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвTensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ┐
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ь
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:В
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*4
_output_shapes"
 :                  К
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:                  К
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : з
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:М
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Е
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  @l
IdentityIdentityBiasAdd:output:0^NoOp*
T0*4
_output_shapes"
 :                  @z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:                  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
°
Х
,__inference_sequential_1_layer_call_fn_20081

inputs
unknown:`
	unknown_0:`
identityИвStatefulPartitionedCallщ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  `*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_15620|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  ``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:                  `: : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :                  `
 
_user_specified_nameinputs
Р%
є
P__inference_layer_normalization_5_layer_call_and_return_conditional_losses_15613

inputs+
mul_3_readvariableop_resource:`)
add_readvariableop_resource:`
identityИвadd/ReadVariableOpвmul_3/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/xConst*
_output_shapes
: *
dtype0*
value	B :S
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: _
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
mul_1Mulmul:z:0strided_slice_1:output:0*
T0*
_output_shapes
: _
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_2Mulmul_2/x:output:0strided_slice_2:output:0*
T0*
_output_shapes
: Q
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Н
Reshape/shapePackReshape/shape/0:output:0	mul_1:z:0	mul_2:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:         `L
ones/packedPack	mul_1:z:0*
N*
T0*
_output_shapes
:O

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?e
onesFillones/packed:output:0ones/Const:output:0*
T0*#
_output_shapes
:         M
zeros/packedPack	mul_1:z:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    h
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*#
_output_shapes
:         H
ConstConst*
_output_shapes
: *
dtype0*
valueB J
Const_1Const*
_output_shapes
: *
dtype0*
valueB г
FusedBatchNormV3FusedBatchNormV3Reshape:output:0ones:output:0zeros:output:0Const:output:0Const_1:output:0*
T0*
U0*o
_output_shapes]
[:         `:         :         :         :         :*
data_formatNCHW*
epsilon%oГ:y
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*4
_output_shapes"
 :                  `n
mul_3/ReadVariableOpReadVariableOpmul_3_readvariableop_resource*
_output_shapes
:`*
dtype0}
mul_3MulReshape_1:output:0mul_3/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:`*
dtype0r
addAddV2	mul_3:z:0add/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `c
IdentityIdentityadd:z:0^NoOp*
T0*4
_output_shapes"
 :                  `r
NoOpNoOp^add/ReadVariableOp^mul_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:                  `: : 2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_3/ReadVariableOpmul_3/ReadVariableOp:\ X
4
_output_shapes"
 :                  `
 
_user_specified_nameinputs
й
├
mpn_scan_while_cond_17976.
*mpn_scan_while_mpn_scan_while_loop_counter)
%mpn_scan_while_mpn_scan_strided_slice
mpn_scan_while_placeholder 
mpn_scan_while_placeholder_1 
mpn_scan_while_placeholder_2.
*mpn_scan_while_less_mpn_scan_strided_sliceE
Ampn_scan_while_mpn_scan_while_cond_17976___redundant_placeholder0E
Ampn_scan_while_mpn_scan_while_cond_17976___redundant_placeholder1E
Ampn_scan_while_mpn_scan_while_cond_17976___redundant_placeholder2E
Ampn_scan_while_mpn_scan_while_cond_17976___redundant_placeholder3
mpn_scan_while_identity
Д
mpn/scan/while/LessLessmpn_scan_while_placeholder*mpn_scan_while_less_mpn_scan_strided_slice*
T0*
_output_shapes
: С
mpn/scan/while/Less_1Less*mpn_scan_while_mpn_scan_while_loop_counter%mpn_scan_while_mpn_scan_strided_slice*
T0*
_output_shapes
: s
mpn/scan/while/LogicalAnd
LogicalAndmpn/scan/while/Less_1:z:0mpn/scan/while/Less:z:0*
_output_shapes
: c
mpn/scan/while/IdentityIdentitympn/scan/while/LogicalAnd:z:0*
T0
*
_output_shapes
: ";
mpn_scan_while_identity mpn/scan/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : :         `: : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         `:

_output_shapes
: :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
:
█
Є
G__inference_sequential_1_layer_call_and_return_conditional_losses_15684
lambda_input)
layer_normalization_5_15678:`)
layer_normalization_5_15680:`
identityИв-layer_normalization_5/StatefulPartitionedCall╞
lambda/PartitionedCallPartitionedCalllambda_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  `* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_lambda_layer_call_and_return_conditional_losses_15364╟
-layer_normalization_5/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0layer_normalization_5_15678layer_normalization_5_15680*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  `*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_layer_normalization_5_layer_call_and_return_conditional_losses_15613Т
IdentityIdentity6layer_normalization_5/StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  `v
NoOpNoOp.^layer_normalization_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:                  `: : 2^
-layer_normalization_5/StatefulPartitionedCall-layer_normalization_5/StatefulPartitionedCall:b ^
4
_output_shapes"
 :                  `
&
_user_specified_namelambda_input
Р%
є
P__inference_layer_normalization_3_layer_call_and_return_conditional_losses_19215

inputs+
mul_3_readvariableop_resource:`)
add_readvariableop_resource:`
identityИвadd/ReadVariableOpвmul_3/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/xConst*
_output_shapes
: *
dtype0*
value	B :S
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: _
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
mul_1Mulmul:z:0strided_slice_1:output:0*
T0*
_output_shapes
: _
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_2Mulmul_2/x:output:0strided_slice_2:output:0*
T0*
_output_shapes
: Q
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Н
Reshape/shapePackReshape/shape/0:output:0	mul_1:z:0	mul_2:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:         `L
ones/packedPack	mul_1:z:0*
N*
T0*
_output_shapes
:O

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?e
onesFillones/packed:output:0ones/Const:output:0*
T0*#
_output_shapes
:         M
zeros/packedPack	mul_1:z:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    h
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*#
_output_shapes
:         H
ConstConst*
_output_shapes
: *
dtype0*
valueB J
Const_1Const*
_output_shapes
: *
dtype0*
valueB г
FusedBatchNormV3FusedBatchNormV3Reshape:output:0ones:output:0zeros:output:0Const:output:0Const_1:output:0*
T0*
U0*o
_output_shapes]
[:         `:         :         :         :         :*
data_formatNCHW*
epsilon%oГ:y
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*4
_output_shapes"
 :                  `n
mul_3/ReadVariableOpReadVariableOpmul_3_readvariableop_resource*
_output_shapes
:`*
dtype0}
mul_3MulReshape_1:output:0mul_3/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:`*
dtype0r
addAddV2	mul_3:z:0add/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `c
IdentityIdentityadd:z:0^NoOp*
T0*4
_output_shapes"
 :                  `r
NoOpNoOp^add/ReadVariableOp^mul_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:                  `: : 2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_3/ReadVariableOpmul_3/ReadVariableOp:\ X
4
_output_shapes"
 :                  `
 
_user_specified_nameinputs
▀;
┬
G__inference_sequential_1_layer_call_and_return_conditional_losses_20200

inputsA
3layer_normalization_5_mul_3_readvariableop_resource:`?
1layer_normalization_5_add_readvariableop_resource:`
identityИв(layer_normalization_5/add/ReadVariableOpв*layer_normalization_5/mul_3/ReadVariableOpV
lambda/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?y
lambda/Gelu/mulMullambda/Gelu/mul/x:output:0inputs*
T0*4
_output_shapes"
 :                  `W
lambda/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *є╡?В
lambda/Gelu/truedivRealDivinputslambda/Gelu/Cast/x:output:0*
T0*4
_output_shapes"
 :                  `n
lambda/Gelu/ErfErflambda/Gelu/truediv:z:0*
T0*4
_output_shapes"
 :                  `V
lambda/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?И
lambda/Gelu/addAddV2lambda/Gelu/add/x:output:0lambda/Gelu/Erf:y:0*
T0*4
_output_shapes"
 :                  `Б
lambda/Gelu/mul_1Mullambda/Gelu/mul:z:0lambda/Gelu/add:z:0*
T0*4
_output_shapes"
 :                  ``
layer_normalization_5/ShapeShapelambda/Gelu/mul_1:z:0*
T0*
_output_shapes
:s
)layer_normalization_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+layer_normalization_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┐
#layer_normalization_5/strided_sliceStridedSlice$layer_normalization_5/Shape:output:02layer_normalization_5/strided_slice/stack:output:04layer_normalization_5/strided_slice/stack_1:output:04layer_normalization_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization_5/mul/xConst*
_output_shapes
: *
dtype0*
value	B :Х
layer_normalization_5/mulMul$layer_normalization_5/mul/x:output:0,layer_normalization_5/strided_slice:output:0*
T0*
_output_shapes
: u
+layer_normalization_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╟
%layer_normalization_5/strided_slice_1StridedSlice$layer_normalization_5/Shape:output:04layer_normalization_5/strided_slice_1/stack:output:06layer_normalization_5/strided_slice_1/stack_1:output:06layer_normalization_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskТ
layer_normalization_5/mul_1Mullayer_normalization_5/mul:z:0.layer_normalization_5/strided_slice_1:output:0*
T0*
_output_shapes
: u
+layer_normalization_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╟
%layer_normalization_5/strided_slice_2StridedSlice$layer_normalization_5/Shape:output:04layer_normalization_5/strided_slice_2/stack:output:06layer_normalization_5/strided_slice_2/stack_1:output:06layer_normalization_5/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
layer_normalization_5/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :Ы
layer_normalization_5/mul_2Mul&layer_normalization_5/mul_2/x:output:0.layer_normalization_5/strided_slice_2:output:0*
T0*
_output_shapes
: g
%layer_normalization_5/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :g
%layer_normalization_5/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :√
#layer_normalization_5/Reshape/shapePack.layer_normalization_5/Reshape/shape/0:output:0layer_normalization_5/mul_1:z:0layer_normalization_5/mul_2:z:0.layer_normalization_5/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:з
layer_normalization_5/ReshapeReshapelambda/Gelu/mul_1:z:0,layer_normalization_5/Reshape/shape:output:0*
T0*/
_output_shapes
:         `x
!layer_normalization_5/ones/packedPacklayer_normalization_5/mul_1:z:0*
N*
T0*
_output_shapes
:e
 layer_normalization_5/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?з
layer_normalization_5/onesFill*layer_normalization_5/ones/packed:output:0)layer_normalization_5/ones/Const:output:0*
T0*#
_output_shapes
:         y
"layer_normalization_5/zeros/packedPacklayer_normalization_5/mul_1:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization_5/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    к
layer_normalization_5/zerosFill+layer_normalization_5/zeros/packed:output:0*layer_normalization_5/zeros/Const:output:0*
T0*#
_output_shapes
:         ^
layer_normalization_5/ConstConst*
_output_shapes
: *
dtype0*
valueB `
layer_normalization_5/Const_1Const*
_output_shapes
: *
dtype0*
valueB з
&layer_normalization_5/FusedBatchNormV3FusedBatchNormV3&layer_normalization_5/Reshape:output:0#layer_normalization_5/ones:output:0$layer_normalization_5/zeros:output:0$layer_normalization_5/Const:output:0&layer_normalization_5/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:         `:         :         :         :         :*
data_formatNCHW*
epsilon%oГ:╗
layer_normalization_5/Reshape_1Reshape*layer_normalization_5/FusedBatchNormV3:y:0$layer_normalization_5/Shape:output:0*
T0*4
_output_shapes"
 :                  `Ъ
*layer_normalization_5/mul_3/ReadVariableOpReadVariableOp3layer_normalization_5_mul_3_readvariableop_resource*
_output_shapes
:`*
dtype0┐
layer_normalization_5/mul_3Mul(layer_normalization_5/Reshape_1:output:02layer_normalization_5/mul_3/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `Ц
(layer_normalization_5/add/ReadVariableOpReadVariableOp1layer_normalization_5_add_readvariableop_resource*
_output_shapes
:`*
dtype0┤
layer_normalization_5/addAddV2layer_normalization_5/mul_3:z:00layer_normalization_5/add/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `y
IdentityIdentitylayer_normalization_5/add:z:0^NoOp*
T0*4
_output_shapes"
 :                  `Ю
NoOpNoOp)^layer_normalization_5/add/ReadVariableOp+^layer_normalization_5/mul_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:                  `: : 2T
(layer_normalization_5/add/ReadVariableOp(layer_normalization_5/add/ReadVariableOp2X
*layer_normalization_5/mul_3/ReadVariableOp*layer_normalization_5/mul_3/ReadVariableOp:\ X
4
_output_shapes"
 :                  `
 
_user_specified_nameinputs
Ж
Ь
3__inference_layer_normalization_layer_call_fn_18883

inputs
unknown:@
	unknown_0:@
identityИвStatefulPartitionedCallЁ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *W
fRRP
N__inference_layer_normalization_layer_call_and_return_conditional_losses_15803|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:                  @: : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :                  @
 
_user_specified_nameinputs
нА
Г	
>__inference_mpn_layer_call_and_return_conditional_losses_16354

inputs
inputs_1
inputs_2
inputs_3
inputs_4E
2sequential_dense_tensordot_readvariableop_resource:	а`>
0sequential_dense_biasadd_readvariableop_resource:`L
>sequential_layer_normalization_4_mul_3_readvariableop_resource:`J
<sequential_layer_normalization_4_add_readvariableop_resource:`<
)dense_1_tensordot_readvariableop_resource:	└`5
'dense_1_biasadd_readvariableop_resource:`N
@sequential_1_layer_normalization_5_mul_3_readvariableop_resource:`L
>sequential_1_layer_normalization_5_add_readvariableop_resource:`
identity

identity_1

identity_2

identity_3

identity_4Ивdense_1/BiasAdd/ReadVariableOpв dense_1/Tensordot/ReadVariableOpв'sequential/dense/BiasAdd/ReadVariableOpв)sequential/dense/Tensordot/ReadVariableOpв3sequential/layer_normalization_4/add/ReadVariableOpв5sequential/layer_normalization_4/mul_3/ReadVariableOpв5sequential_1/layer_normalization_5/add/ReadVariableOpв7sequential_1/layer_normalization_5/mul_3/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Shape_1Shapeinputs_2*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask=
Shape_2Shapeinputs*
T0*
_output_shapes
:_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_2StridedSliceShape_2:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskO
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :╢
GatherV2GatherV2inputsinputs_2GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*8
_output_shapes&
$:"                  `*

batch_dimsR
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :└С
Reshape/shapePackstrided_slice_2:output:0strided_slice_1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:}
ReshapeReshapeGatherV2:output:0Reshape/shape:output:0*
T0*5
_output_shapes#
!:                  └V
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
         Н
concatConcatV2Reshape:output:0inputs_1concat/axis:output:0*
N*
T0*5
_output_shapes#
!:                  аЭ
)sequential/dense/Tensordot/ReadVariableOpReadVariableOp2sequential_dense_tensordot_readvariableop_resource*
_output_shapes
:	а`*
dtype0i
sequential/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:p
sequential/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       _
 sequential/dense/Tensordot/ShapeShapeconcat:output:0*
T0*
_output_shapes
:j
(sequential/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :  
#sequential/dense/Tensordot/GatherV2GatherV2)sequential/dense/Tensordot/Shape:output:0(sequential/dense/Tensordot/free:output:01sequential/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
*sequential/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Г
%sequential/dense/Tensordot/GatherV2_1GatherV2)sequential/dense/Tensordot/Shape:output:0(sequential/dense/Tensordot/axes:output:03sequential/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:j
 sequential/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: б
sequential/dense/Tensordot/ProdProd,sequential/dense/Tensordot/GatherV2:output:0)sequential/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: l
"sequential/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: з
!sequential/dense/Tensordot/Prod_1Prod.sequential/dense/Tensordot/GatherV2_1:output:0+sequential/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: h
&sequential/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : р
!sequential/dense/Tensordot/concatConcatV2(sequential/dense/Tensordot/free:output:0(sequential/dense/Tensordot/axes:output:0/sequential/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:м
 sequential/dense/Tensordot/stackPack(sequential/dense/Tensordot/Prod:output:0*sequential/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:о
$sequential/dense/Tensordot/transpose	Transposeconcat:output:0*sequential/dense/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:                  а╜
"sequential/dense/Tensordot/ReshapeReshape(sequential/dense/Tensordot/transpose:y:0)sequential/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  ╜
!sequential/dense/Tensordot/MatMulMatMul+sequential/dense/Tensordot/Reshape:output:01sequential/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `l
"sequential/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:`j
(sequential/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ы
#sequential/dense/Tensordot/concat_1ConcatV2,sequential/dense/Tensordot/GatherV2:output:0+sequential/dense/Tensordot/Const_2:output:01sequential/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:┐
sequential/dense/TensordotReshape+sequential/dense/Tensordot/MatMul:product:0,sequential/dense/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  `Ф
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0╕
sequential/dense/BiasAddBiasAdd#sequential/dense/Tensordot:output:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `ц
!sequential/lambda/PartitionedCallPartitionedCall!sequential/dense/BiasAdd:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  `* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_lambda_layer_call_and_return_conditional_losses_15364А
&sequential/layer_normalization_4/ShapeShape*sequential/lambda/PartitionedCall:output:0*
T0*
_output_shapes
:~
4sequential/layer_normalization_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: А
6sequential/layer_normalization_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:А
6sequential/layer_normalization_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ў
.sequential/layer_normalization_4/strided_sliceStridedSlice/sequential/layer_normalization_4/Shape:output:0=sequential/layer_normalization_4/strided_slice/stack:output:0?sequential/layer_normalization_4/strided_slice/stack_1:output:0?sequential/layer_normalization_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
&sequential/layer_normalization_4/mul/xConst*
_output_shapes
: *
dtype0*
value	B :╢
$sequential/layer_normalization_4/mulMul/sequential/layer_normalization_4/mul/x:output:07sequential/layer_normalization_4/strided_slice:output:0*
T0*
_output_shapes
: А
6sequential/layer_normalization_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:В
8sequential/layer_normalization_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:В
8sequential/layer_normalization_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:■
0sequential/layer_normalization_4/strided_slice_1StridedSlice/sequential/layer_normalization_4/Shape:output:0?sequential/layer_normalization_4/strided_slice_1/stack:output:0Asequential/layer_normalization_4/strided_slice_1/stack_1:output:0Asequential/layer_normalization_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask│
&sequential/layer_normalization_4/mul_1Mul(sequential/layer_normalization_4/mul:z:09sequential/layer_normalization_4/strided_slice_1:output:0*
T0*
_output_shapes
: А
6sequential/layer_normalization_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:В
8sequential/layer_normalization_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:В
8sequential/layer_normalization_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:■
0sequential/layer_normalization_4/strided_slice_2StridedSlice/sequential/layer_normalization_4/Shape:output:0?sequential/layer_normalization_4/strided_slice_2/stack:output:0Asequential/layer_normalization_4/strided_slice_2/stack_1:output:0Asequential/layer_normalization_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(sequential/layer_normalization_4/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :╝
&sequential/layer_normalization_4/mul_2Mul1sequential/layer_normalization_4/mul_2/x:output:09sequential/layer_normalization_4/strided_slice_2:output:0*
T0*
_output_shapes
: r
0sequential/layer_normalization_4/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :r
0sequential/layer_normalization_4/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :▓
.sequential/layer_normalization_4/Reshape/shapePack9sequential/layer_normalization_4/Reshape/shape/0:output:0*sequential/layer_normalization_4/mul_1:z:0*sequential/layer_normalization_4/mul_2:z:09sequential/layer_normalization_4/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:╥
(sequential/layer_normalization_4/ReshapeReshape*sequential/lambda/PartitionedCall:output:07sequential/layer_normalization_4/Reshape/shape:output:0*
T0*/
_output_shapes
:         `О
,sequential/layer_normalization_4/ones/packedPack*sequential/layer_normalization_4/mul_1:z:0*
N*
T0*
_output_shapes
:p
+sequential/layer_normalization_4/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╚
%sequential/layer_normalization_4/onesFill5sequential/layer_normalization_4/ones/packed:output:04sequential/layer_normalization_4/ones/Const:output:0*
T0*#
_output_shapes
:         П
-sequential/layer_normalization_4/zeros/packedPack*sequential/layer_normalization_4/mul_1:z:0*
N*
T0*
_output_shapes
:q
,sequential/layer_normalization_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ╦
&sequential/layer_normalization_4/zerosFill6sequential/layer_normalization_4/zeros/packed:output:05sequential/layer_normalization_4/zeros/Const:output:0*
T0*#
_output_shapes
:         i
&sequential/layer_normalization_4/ConstConst*
_output_shapes
: *
dtype0*
valueB k
(sequential/layer_normalization_4/Const_1Const*
_output_shapes
: *
dtype0*
valueB щ
1sequential/layer_normalization_4/FusedBatchNormV3FusedBatchNormV31sequential/layer_normalization_4/Reshape:output:0.sequential/layer_normalization_4/ones:output:0/sequential/layer_normalization_4/zeros:output:0/sequential/layer_normalization_4/Const:output:01sequential/layer_normalization_4/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:         `:         :         :         :         :*
data_formatNCHW*
epsilon%oГ:▄
*sequential/layer_normalization_4/Reshape_1Reshape5sequential/layer_normalization_4/FusedBatchNormV3:y:0/sequential/layer_normalization_4/Shape:output:0*
T0*4
_output_shapes"
 :                  `░
5sequential/layer_normalization_4/mul_3/ReadVariableOpReadVariableOp>sequential_layer_normalization_4_mul_3_readvariableop_resource*
_output_shapes
:`*
dtype0р
&sequential/layer_normalization_4/mul_3Mul3sequential/layer_normalization_4/Reshape_1:output:0=sequential/layer_normalization_4/mul_3/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `м
3sequential/layer_normalization_4/add/ReadVariableOpReadVariableOp<sequential_layer_normalization_4_add_readvariableop_resource*
_output_shapes
:`*
dtype0╒
$sequential/layer_normalization_4/addAddV2*sequential/layer_normalization_4/mul_3:z:0;sequential/layer_normalization_4/add/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `}
mulMul(sequential/layer_normalization_4/add:z:0inputs_3*
T0*4
_output_shapes"
 :                  `P
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :`s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         `A

scan/ShapeShapemul:z:0*
T0*
_output_shapes
:b
scan/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: d
scan/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:d
scan/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ъ
scan/strided_sliceStridedSlicescan/Shape:output:0!scan/strided_slice/stack:output:0#scan/strided_slice/stack_1:output:0#scan/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskq
 scan/TensorArrayV2/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    `   ┴
scan/TensorArrayV2TensorListReserve)scan/TensorArrayV2/element_shape:output:0scan/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥s
"scan/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ┼
scan/TensorArrayV2_1TensorListReserve+scan/TensorArrayV2_1/element_shape:output:0scan/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╧s
"scan/TensorArrayV2_2/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ┼
scan/TensorArrayV2_2TensorListReserve+scan/TensorArrayV2_2/element_shape:output:0scan/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Л
:scan/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    `   ф
,scan/TensorArrayUnstack/TensorListFromTensorTensorListFromTensormul:z:0Cscan/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Н
<scan/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       щ
.scan/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensorinputs_2Escan/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╧Н
<scan/TensorArrayUnstack_2/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       щ
.scan/TensorArrayUnstack_2/TensorListFromTensorTensorListFromTensorinputs_4Escan/TensorArrayUnstack_2/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥s
"scan/TensorArrayV2_3/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    `   ┼
scan/TensorArrayV2_3TensorListReserve+scan/TensorArrayV2_3/element_shape:output:0scan/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥L

scan/ConstConst*
_output_shapes
: *
dtype0*
value	B : Y
scan/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ▀

scan/whileStatelessWhile scan/while/loop_counter:output:0scan/strided_slice:output:0scan/Const:output:0zeros:output:0scan/TensorArrayV2_3:handle:0scan/strided_slice:output:0<scan/TensorArrayUnstack/TensorListFromTensor:output_handle:0>scan/TensorArrayUnstack_1/TensorListFromTensor:output_handle:0>scan/TensorArrayUnstack_2/TensorListFromTensor:output_handle:0strided_slice:output:0*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : :         `: : : : : : * 
_read_only_resource_inputs
 *
_stateful_parallelism( *!
bodyR
scan_while_body_16211*!
condR
scan_while_cond_16210*8
output_shapes'
%: : : :         `: : : : : : Ж
5scan/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    `   ┌
'scan/TensorArrayV2Stack/TensorListStackTensorListStackscan/while:output:4>scan/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  `*
element_dtype0_
lambda_1/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
         ╜
lambda_1/concatConcatV2inputs0scan/TensorArrayV2Stack/TensorListStack:tensor:0lambda_1/concat/axis:output:0*
N*
T0*5
_output_shapes#
!:                  └Л
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource*
_output_shapes
:	└`*
dtype0`
dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       _
dense_1/Tensordot/ShapeShapelambda_1/concat:output:0*
T0*
_output_shapes
:a
dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : █
dense_1/Tensordot/GatherV2GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/free:output:0(dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ▀
dense_1/Tensordot/GatherV2_1GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/axes:output:0*dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: Ж
dense_1/Tensordot/ProdProd#dense_1/Tensordot/GatherV2:output:0 dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: М
dense_1/Tensordot/Prod_1Prod%dense_1/Tensordot/GatherV2_1:output:0"dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ╝
dense_1/Tensordot/concatConcatV2dense_1/Tensordot/free:output:0dense_1/Tensordot/axes:output:0&dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:С
dense_1/Tensordot/stackPackdense_1/Tensordot/Prod:output:0!dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:е
dense_1/Tensordot/transpose	Transposelambda_1/concat:output:0!dense_1/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:                  └в
dense_1/Tensordot/ReshapeReshapedense_1/Tensordot/transpose:y:0 dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  в
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0(dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `c
dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:`a
dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ╟
dense_1/Tensordot/concat_1ConcatV2#dense_1/Tensordot/GatherV2:output:0"dense_1/Tensordot/Const_2:output:0(dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:д
dense_1/TensordotReshape"dense_1/Tensordot/MatMul:product:0#dense_1/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  `В
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0Э
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `▀
#sequential_1/lambda/PartitionedCallPartitionedCalldense_1/BiasAdd:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  `* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_lambda_layer_call_and_return_conditional_losses_15364Д
(sequential_1/layer_normalization_5/ShapeShape,sequential_1/lambda/PartitionedCall:output:0*
T0*
_output_shapes
:А
6sequential_1/layer_normalization_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: В
8sequential_1/layer_normalization_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:В
8sequential_1/layer_normalization_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:А
0sequential_1/layer_normalization_5/strided_sliceStridedSlice1sequential_1/layer_normalization_5/Shape:output:0?sequential_1/layer_normalization_5/strided_slice/stack:output:0Asequential_1/layer_normalization_5/strided_slice/stack_1:output:0Asequential_1/layer_normalization_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(sequential_1/layer_normalization_5/mul/xConst*
_output_shapes
: *
dtype0*
value	B :╝
&sequential_1/layer_normalization_5/mulMul1sequential_1/layer_normalization_5/mul/x:output:09sequential_1/layer_normalization_5/strided_slice:output:0*
T0*
_output_shapes
: В
8sequential_1/layer_normalization_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:Д
:sequential_1/layer_normalization_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Д
:sequential_1/layer_normalization_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
2sequential_1/layer_normalization_5/strided_slice_1StridedSlice1sequential_1/layer_normalization_5/Shape:output:0Asequential_1/layer_normalization_5/strided_slice_1/stack:output:0Csequential_1/layer_normalization_5/strided_slice_1/stack_1:output:0Csequential_1/layer_normalization_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask╣
(sequential_1/layer_normalization_5/mul_1Mul*sequential_1/layer_normalization_5/mul:z:0;sequential_1/layer_normalization_5/strided_slice_1:output:0*
T0*
_output_shapes
: В
8sequential_1/layer_normalization_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:Д
:sequential_1/layer_normalization_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Д
:sequential_1/layer_normalization_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
2sequential_1/layer_normalization_5/strided_slice_2StridedSlice1sequential_1/layer_normalization_5/Shape:output:0Asequential_1/layer_normalization_5/strided_slice_2/stack:output:0Csequential_1/layer_normalization_5/strided_slice_2/stack_1:output:0Csequential_1/layer_normalization_5/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
*sequential_1/layer_normalization_5/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :┬
(sequential_1/layer_normalization_5/mul_2Mul3sequential_1/layer_normalization_5/mul_2/x:output:0;sequential_1/layer_normalization_5/strided_slice_2:output:0*
T0*
_output_shapes
: t
2sequential_1/layer_normalization_5/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :t
2sequential_1/layer_normalization_5/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :╝
0sequential_1/layer_normalization_5/Reshape/shapePack;sequential_1/layer_normalization_5/Reshape/shape/0:output:0,sequential_1/layer_normalization_5/mul_1:z:0,sequential_1/layer_normalization_5/mul_2:z:0;sequential_1/layer_normalization_5/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:╪
*sequential_1/layer_normalization_5/ReshapeReshape,sequential_1/lambda/PartitionedCall:output:09sequential_1/layer_normalization_5/Reshape/shape:output:0*
T0*/
_output_shapes
:         `Т
.sequential_1/layer_normalization_5/ones/packedPack,sequential_1/layer_normalization_5/mul_1:z:0*
N*
T0*
_output_shapes
:r
-sequential_1/layer_normalization_5/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╬
'sequential_1/layer_normalization_5/onesFill7sequential_1/layer_normalization_5/ones/packed:output:06sequential_1/layer_normalization_5/ones/Const:output:0*
T0*#
_output_shapes
:         У
/sequential_1/layer_normalization_5/zeros/packedPack,sequential_1/layer_normalization_5/mul_1:z:0*
N*
T0*
_output_shapes
:s
.sequential_1/layer_normalization_5/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ╤
(sequential_1/layer_normalization_5/zerosFill8sequential_1/layer_normalization_5/zeros/packed:output:07sequential_1/layer_normalization_5/zeros/Const:output:0*
T0*#
_output_shapes
:         k
(sequential_1/layer_normalization_5/ConstConst*
_output_shapes
: *
dtype0*
valueB m
*sequential_1/layer_normalization_5/Const_1Const*
_output_shapes
: *
dtype0*
valueB ї
3sequential_1/layer_normalization_5/FusedBatchNormV3FusedBatchNormV33sequential_1/layer_normalization_5/Reshape:output:00sequential_1/layer_normalization_5/ones:output:01sequential_1/layer_normalization_5/zeros:output:01sequential_1/layer_normalization_5/Const:output:03sequential_1/layer_normalization_5/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:         `:         :         :         :         :*
data_formatNCHW*
epsilon%oГ:т
,sequential_1/layer_normalization_5/Reshape_1Reshape7sequential_1/layer_normalization_5/FusedBatchNormV3:y:01sequential_1/layer_normalization_5/Shape:output:0*
T0*4
_output_shapes"
 :                  `┤
7sequential_1/layer_normalization_5/mul_3/ReadVariableOpReadVariableOp@sequential_1_layer_normalization_5_mul_3_readvariableop_resource*
_output_shapes
:`*
dtype0ц
(sequential_1/layer_normalization_5/mul_3Mul5sequential_1/layer_normalization_5/Reshape_1:output:0?sequential_1/layer_normalization_5/mul_3/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `░
5sequential_1/layer_normalization_5/add/ReadVariableOpReadVariableOp>sequential_1_layer_normalization_5_add_readvariableop_resource*
_output_shapes
:`*
dtype0█
&sequential_1/layer_normalization_5/addAddV2,sequential_1/layer_normalization_5/mul_3:z:0=sequential_1/layer_normalization_5/add/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `Ж
IdentityIdentity*sequential_1/layer_normalization_5/add:z:0^NoOp*
T0*4
_output_shapes"
 :                  `e

Identity_1Identitymul:z:0^NoOp*
T0*4
_output_shapes"
 :                  `f

Identity_2Identityinputs_2^NoOp*
T0*4
_output_shapes"
 :                  f

Identity_3Identityinputs_3^NoOp*
T0*4
_output_shapes"
 :                  f

Identity_4Identityinputs_4^NoOp*
T0*4
_output_shapes"
 :                  └
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp*^sequential/dense/Tensordot/ReadVariableOp4^sequential/layer_normalization_4/add/ReadVariableOp6^sequential/layer_normalization_4/mul_3/ReadVariableOp6^sequential_1/layer_normalization_5/add/ReadVariableOp8^sequential_1/layer_normalization_5/mul_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*┼
_input_shapes│
░:                  `:                  `:                  :                  :                  : : : : : : : : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_1/Tensordot/ReadVariableOp dense_1/Tensordot/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2V
)sequential/dense/Tensordot/ReadVariableOp)sequential/dense/Tensordot/ReadVariableOp2j
3sequential/layer_normalization_4/add/ReadVariableOp3sequential/layer_normalization_4/add/ReadVariableOp2n
5sequential/layer_normalization_4/mul_3/ReadVariableOp5sequential/layer_normalization_4/mul_3/ReadVariableOp2n
5sequential_1/layer_normalization_5/add/ReadVariableOp5sequential_1/layer_normalization_5/add/ReadVariableOp2r
7sequential_1/layer_normalization_5/mul_3/ReadVariableOp7sequential_1/layer_normalization_5/mul_3/ReadVariableOp:\ X
4
_output_shapes"
 :                  `
 
_user_specified_nameinputs:\X
4
_output_shapes"
 :                  `
 
_user_specified_nameinputs:\X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs:\X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs:\X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
Г
√
D__inference_node_ide2_layer_call_and_return_conditional_losses_18969

inputs3
!tensordot_readvariableop_resource:@`-
biasadd_readvariableop_resource:`
identityИвBiasAdd/ReadVariableOpвTensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@`*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ┐
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ь
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:В
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*4
_output_shapes"
 :                  @К
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:                  К
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:`Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : з
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:М
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  `r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:`*
dtype0Е
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `l
IdentityIdentityBiasAdd:output:0^NoOp*
T0*4
_output_shapes"
 :                  `z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:                  @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:\ X
4
_output_shapes"
 :                  @
 
_user_specified_nameinputs
тS
Щ
@__inference_model_layer_call_and_return_conditional_losses_17347
input_1
input_2
input_3
input_4!
node_ide1_17269:@
node_ide1_17271:@'
layer_normalization_17275:@'
layer_normalization_17277:@!
edge_ide1_17280:@
edge_ide1_17282:@!
node_ide2_17285:@`
node_ide2_17287:`)
layer_normalization_1_17292:@)
layer_normalization_1_17294:@!
edge_ide2_17297:@`
edge_ide2_17299:`)
layer_normalization_2_17307:`)
layer_normalization_2_17309:`)
layer_normalization_3_17312:`)
layer_normalization_3_17314:`
	mpn_17320:	а`
	mpn_17322:`
	mpn_17324:`
	mpn_17326:`
	mpn_17328:	└`
	mpn_17330:`
	mpn_17332:`
	mpn_17334:`'
node_prediction_17341:`#
node_prediction_17343:
identityИв!edge_ide1/StatefulPartitionedCallв!edge_ide2/StatefulPartitionedCallв+layer_normalization/StatefulPartitionedCallв-layer_normalization_1/StatefulPartitionedCallв-layer_normalization_2/StatefulPartitionedCallв-layer_normalization_3/StatefulPartitionedCallвmpn/StatefulPartitionedCallв!node_ide1/StatefulPartitionedCallв!node_ide2/StatefulPartitionedCallв'node_prediction/StatefulPartitionedCall 
!node_ide1/StatefulPartitionedCallStatefulPartitionedCallinput_1node_ide1_17269node_ide1_17271*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_node_ide1_layer_call_and_return_conditional_losses_15737ф
lambda/PartitionedCallPartitionedCall*node_ide1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_lambda_layer_call_and_return_conditional_losses_16872┐
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0layer_normalization_17275layer_normalization_17277*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *W
fRRP
N__inference_layer_normalization_layer_call_and_return_conditional_losses_15803 
!edge_ide1/StatefulPartitionedCallStatefulPartitionedCallinput_2edge_ide1_17280edge_ide1_17282*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_edge_ide1_layer_call_and_return_conditional_losses_15839м
!node_ide2/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0node_ide2_17285node_ide2_17287*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  `*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_node_ide2_layer_call_and_return_conditional_losses_15875ц
lambda/PartitionedCall_1PartitionedCall*edge_ide1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_lambda_layer_call_and_return_conditional_losses_16872ц
lambda/PartitionedCall_2PartitionedCall*node_ide2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  `* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_lambda_layer_call_and_return_conditional_losses_15461╔
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCall!lambda/PartitionedCall_1:output:0layer_normalization_1_17292layer_normalization_1_17294*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_layer_normalization_1_layer_call_and_return_conditional_losses_15930о
!edge_ide2/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_1/StatefulPartitionedCall:output:0edge_ide2_17297edge_ide2_17299*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  `*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_edge_ide2_layer_call_and_return_conditional_losses_15966}
,tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
.tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
.tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╔
&tf.__operators__.getitem/strided_sliceStridedSliceinput_25tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*4
_output_shapes"
 :                  *
ellipsis_maskц
lambda/PartitionedCall_3PartitionedCall*edge_ide2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  `* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_lambda_layer_call_and_return_conditional_losses_15461╔
-layer_normalization_2/StatefulPartitionedCallStatefulPartitionedCall!lambda/PartitionedCall_2:output:0layer_normalization_2_17307layer_normalization_2_17309*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  `*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_layer_normalization_2_layer_call_and_return_conditional_losses_16024╔
-layer_normalization_3/StatefulPartitionedCallStatefulPartitionedCall!lambda/PartitionedCall_3:output:0layer_normalization_3_17312layer_normalization_3_17314*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  `*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_layer_normalization_3_layer_call_and_return_conditional_losses_16077{
tf.ones_like/ones_like/ShapeShape/tf.__operators__.getitem/strided_slice:output:0*
T0*
_output_shapes
:a
tf.ones_like/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?л
tf.ones_like/ones_likeFill%tf.ones_like/ones_like/Shape:output:0%tf.ones_like/ones_like/Const:output:0*
T0*4
_output_shapes"
 :                  ┌
mpn/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_2/StatefulPartitionedCall:output:06layer_normalization_3/StatefulPartitionedCall:output:0input_3tf.ones_like/ones_like:output:0input_4	mpn_17320	mpn_17322	mpn_17324	mpn_17326	mpn_17328	mpn_17330	mpn_17332	mpn_17334*
Tin
2*
Tout	
2*
_collective_manager_ids
 *╢
_output_shapesг
а:                  `:                  `:                  :                  :                  **
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8В *G
fBR@
>__inference_mpn_layer_call_and_return_conditional_losses_16786┤
'node_prediction/StatefulPartitionedCallStatefulPartitionedCall$mpn/StatefulPartitionedCall:output:0node_prediction_17341node_prediction_17343*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_node_prediction_layer_call_and_return_conditional_losses_16406М
IdentityIdentity0node_prediction/StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  ▄
NoOpNoOp"^edge_ide1/StatefulPartitionedCall"^edge_ide2/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall.^layer_normalization_1/StatefulPartitionedCall.^layer_normalization_2/StatefulPartitionedCall.^layer_normalization_3/StatefulPartitionedCall^mpn/StatefulPartitionedCall"^node_ide1/StatefulPartitionedCall"^node_ide2/StatefulPartitionedCall(^node_prediction/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*╔
_input_shapes╖
┤:                  :                  :                  :                  : : : : : : : : : : : : : : : : : : : : : : : : : : 2F
!edge_ide1/StatefulPartitionedCall!edge_ide1/StatefulPartitionedCall2F
!edge_ide2/StatefulPartitionedCall!edge_ide2/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2^
-layer_normalization_1/StatefulPartitionedCall-layer_normalization_1/StatefulPartitionedCall2^
-layer_normalization_2/StatefulPartitionedCall-layer_normalization_2/StatefulPartitionedCall2^
-layer_normalization_3/StatefulPartitionedCall-layer_normalization_3/StatefulPartitionedCall2:
mpn/StatefulPartitionedCallmpn/StatefulPartitionedCall2F
!node_ide1/StatefulPartitionedCall!node_ide1/StatefulPartitionedCall2F
!node_ide2/StatefulPartitionedCall!node_ide2/StatefulPartitionedCall2R
'node_prediction/StatefulPartitionedCall'node_prediction/StatefulPartitionedCall:] Y
4
_output_shapes"
 :                  
!
_user_specified_name	input_1:]Y
4
_output_shapes"
 :                  
!
_user_specified_name	input_2:]Y
4
_output_shapes"
 :                  
!
_user_specified_name	input_3:]Y
4
_output_shapes"
 :                  
!
_user_specified_name	input_4
╝	
]
A__inference_lambda_layer_call_and_return_conditional_losses_18850

inputs
identityO

Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?k
Gelu/mulMulGelu/mul/x:output:0inputs*
T0*4
_output_shapes"
 :                  `P
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *є╡?t
Gelu/truedivRealDivinputsGelu/Cast/x:output:0*
T0*4
_output_shapes"
 :                  ``
Gelu/ErfErfGelu/truediv:z:0*
T0*4
_output_shapes"
 :                  `O

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?s
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*4
_output_shapes"
 :                  `l

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*4
_output_shapes"
 :                  `c
IdentityIdentityGelu/mul_1:z:0*
T0*4
_output_shapes"
 :                  `"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :                  `:\ X
4
_output_shapes"
 :                  `
 
_user_specified_nameinputs
╦
╩
*__inference_sequential_layer_call_fn_19910

inputs
unknown:	а`
	unknown_0:`
	unknown_1:`
	unknown_2:`
identityИвStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  `*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_15504|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  ``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):                  а: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:                  а
 
_user_specified_nameinputs
Р%
є
P__inference_layer_normalization_5_layer_call_and_return_conditional_losses_20351

inputs+
mul_3_readvariableop_resource:`)
add_readvariableop_resource:`
identityИвadd/ReadVariableOpвmul_3/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/xConst*
_output_shapes
: *
dtype0*
value	B :S
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: _
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
mul_1Mulmul:z:0strided_slice_1:output:0*
T0*
_output_shapes
: _
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_2Mulmul_2/x:output:0strided_slice_2:output:0*
T0*
_output_shapes
: Q
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Н
Reshape/shapePackReshape/shape/0:output:0	mul_1:z:0	mul_2:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:         `L
ones/packedPack	mul_1:z:0*
N*
T0*
_output_shapes
:O

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?e
onesFillones/packed:output:0ones/Const:output:0*
T0*#
_output_shapes
:         M
zeros/packedPack	mul_1:z:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    h
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*#
_output_shapes
:         H
ConstConst*
_output_shapes
: *
dtype0*
valueB J
Const_1Const*
_output_shapes
: *
dtype0*
valueB г
FusedBatchNormV3FusedBatchNormV3Reshape:output:0ones:output:0zeros:output:0Const:output:0Const_1:output:0*
T0*
U0*o
_output_shapes]
[:         `:         :         :         :         :*
data_formatNCHW*
epsilon%oГ:y
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*4
_output_shapes"
 :                  `n
mul_3/ReadVariableOpReadVariableOpmul_3_readvariableop_resource*
_output_shapes
:`*
dtype0}
mul_3MulReshape_1:output:0mul_3/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:`*
dtype0r
addAddV2	mul_3:z:0add/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `c
IdentityIdentityadd:z:0^NoOp*
T0*4
_output_shapes"
 :                  `r
NoOpNoOp^add/ReadVariableOp^mul_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:                  `: : 2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_3/ReadVariableOpmul_3/ReadVariableOp:\ X
4
_output_shapes"
 :                  `
 
_user_specified_nameinputs
Р%
є
P__inference_layer_normalization_1_layer_call_and_return_conditional_losses_19064

inputs+
mul_3_readvariableop_resource:@)
add_readvariableop_resource:@
identityИвadd/ReadVariableOpвmul_3/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/xConst*
_output_shapes
: *
dtype0*
value	B :S
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: _
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
mul_1Mulmul:z:0strided_slice_1:output:0*
T0*
_output_shapes
: _
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_2Mulmul_2/x:output:0strided_slice_2:output:0*
T0*
_output_shapes
: Q
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Н
Reshape/shapePackReshape/shape/0:output:0	mul_1:z:0	mul_2:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:         @L
ones/packedPack	mul_1:z:0*
N*
T0*
_output_shapes
:O

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?e
onesFillones/packed:output:0ones/Const:output:0*
T0*#
_output_shapes
:         M
zeros/packedPack	mul_1:z:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    h
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*#
_output_shapes
:         H
ConstConst*
_output_shapes
: *
dtype0*
valueB J
Const_1Const*
_output_shapes
: *
dtype0*
valueB г
FusedBatchNormV3FusedBatchNormV3Reshape:output:0ones:output:0zeros:output:0Const:output:0Const_1:output:0*
T0*
U0*o
_output_shapes]
[:         @:         :         :         :         :*
data_formatNCHW*
epsilon%oГ:y
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*4
_output_shapes"
 :                  @n
mul_3/ReadVariableOpReadVariableOpmul_3_readvariableop_resource*
_output_shapes
:@*
dtype0}
mul_3MulReshape_1:output:0mul_3/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  @j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:@*
dtype0r
addAddV2	mul_3:z:0add/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  @c
IdentityIdentityadd:z:0^NoOp*
T0*4
_output_shapes"
 :                  @r
NoOpNoOp^add/ReadVariableOp^mul_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:                  @: : 2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_3/ReadVariableOpmul_3/ReadVariableOp:\ X
4
_output_shapes"
 :                  @
 
_user_specified_nameinputs
╦
B
&__inference_lambda_layer_call_fn_18821

inputs
identity╣
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_lambda_layer_call_and_return_conditional_losses_15754m
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :                  @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :                  @:\ X
4
_output_shapes"
 :                  @
 
_user_specified_nameinputs
ЦШ
╣
@__inference_model_layer_call_and_return_conditional_losses_18767
inputs_0
inputs_1
inputs_2
inputs_3=
+node_ide1_tensordot_readvariableop_resource:@7
)node_ide1_biasadd_readvariableop_resource:@?
1layer_normalization_mul_3_readvariableop_resource:@=
/layer_normalization_add_readvariableop_resource:@=
+edge_ide1_tensordot_readvariableop_resource:@7
)edge_ide1_biasadd_readvariableop_resource:@=
+node_ide2_tensordot_readvariableop_resource:@`7
)node_ide2_biasadd_readvariableop_resource:`A
3layer_normalization_1_mul_3_readvariableop_resource:@?
1layer_normalization_1_add_readvariableop_resource:@=
+edge_ide2_tensordot_readvariableop_resource:@`7
)edge_ide2_biasadd_readvariableop_resource:`A
3layer_normalization_2_mul_3_readvariableop_resource:`?
1layer_normalization_2_add_readvariableop_resource:`A
3layer_normalization_3_mul_3_readvariableop_resource:`?
1layer_normalization_3_add_readvariableop_resource:`I
6mpn_sequential_dense_tensordot_readvariableop_resource:	а`B
4mpn_sequential_dense_biasadd_readvariableop_resource:`P
Bmpn_sequential_layer_normalization_4_mul_3_readvariableop_resource:`N
@mpn_sequential_layer_normalization_4_add_readvariableop_resource:`@
-mpn_dense_1_tensordot_readvariableop_resource:	└`9
+mpn_dense_1_biasadd_readvariableop_resource:`R
Dmpn_sequential_1_layer_normalization_5_mul_3_readvariableop_resource:`P
Bmpn_sequential_1_layer_normalization_5_add_readvariableop_resource:`C
1node_prediction_tensordot_readvariableop_resource:`=
/node_prediction_biasadd_readvariableop_resource:
identityИв edge_ide1/BiasAdd/ReadVariableOpв"edge_ide1/Tensordot/ReadVariableOpв edge_ide2/BiasAdd/ReadVariableOpв"edge_ide2/Tensordot/ReadVariableOpв&layer_normalization/add/ReadVariableOpв(layer_normalization/mul_3/ReadVariableOpв(layer_normalization_1/add/ReadVariableOpв*layer_normalization_1/mul_3/ReadVariableOpв(layer_normalization_2/add/ReadVariableOpв*layer_normalization_2/mul_3/ReadVariableOpв(layer_normalization_3/add/ReadVariableOpв*layer_normalization_3/mul_3/ReadVariableOpв"mpn/dense_1/BiasAdd/ReadVariableOpв$mpn/dense_1/Tensordot/ReadVariableOpв+mpn/sequential/dense/BiasAdd/ReadVariableOpв-mpn/sequential/dense/Tensordot/ReadVariableOpв7mpn/sequential/layer_normalization_4/add/ReadVariableOpв9mpn/sequential/layer_normalization_4/mul_3/ReadVariableOpв9mpn/sequential_1/layer_normalization_5/add/ReadVariableOpв;mpn/sequential_1/layer_normalization_5/mul_3/ReadVariableOpв node_ide1/BiasAdd/ReadVariableOpв"node_ide1/Tensordot/ReadVariableOpв node_ide2/BiasAdd/ReadVariableOpв"node_ide2/Tensordot/ReadVariableOpв&node_prediction/BiasAdd/ReadVariableOpв(node_prediction/Tensordot/ReadVariableOpО
"node_ide1/Tensordot/ReadVariableOpReadVariableOp+node_ide1_tensordot_readvariableop_resource*
_output_shapes

:@*
dtype0b
node_ide1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:i
node_ide1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       Q
node_ide1/Tensordot/ShapeShapeinputs_0*
T0*
_output_shapes
:c
!node_ide1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : у
node_ide1/Tensordot/GatherV2GatherV2"node_ide1/Tensordot/Shape:output:0!node_ide1/Tensordot/free:output:0*node_ide1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:e
#node_ide1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ч
node_ide1/Tensordot/GatherV2_1GatherV2"node_ide1/Tensordot/Shape:output:0!node_ide1/Tensordot/axes:output:0,node_ide1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
node_ide1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: М
node_ide1/Tensordot/ProdProd%node_ide1/Tensordot/GatherV2:output:0"node_ide1/Tensordot/Const:output:0*
T0*
_output_shapes
: e
node_ide1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Т
node_ide1/Tensordot/Prod_1Prod'node_ide1/Tensordot/GatherV2_1:output:0$node_ide1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: a
node_ide1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ─
node_ide1/Tensordot/concatConcatV2!node_ide1/Tensordot/free:output:0!node_ide1/Tensordot/axes:output:0(node_ide1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Ч
node_ide1/Tensordot/stackPack!node_ide1/Tensordot/Prod:output:0#node_ide1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Ш
node_ide1/Tensordot/transpose	Transposeinputs_0#node_ide1/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :                  и
node_ide1/Tensordot/ReshapeReshape!node_ide1/Tensordot/transpose:y:0"node_ide1/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  и
node_ide1/Tensordot/MatMulMatMul$node_ide1/Tensordot/Reshape:output:0*node_ide1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @e
node_ide1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@c
!node_ide1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ╧
node_ide1/Tensordot/concat_1ConcatV2%node_ide1/Tensordot/GatherV2:output:0$node_ide1/Tensordot/Const_2:output:0*node_ide1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:к
node_ide1/TensordotReshape$node_ide1/Tensordot/MatMul:product:0%node_ide1/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  @Ж
 node_ide1/BiasAdd/ReadVariableOpReadVariableOp)node_ide1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0г
node_ide1/BiasAddBiasAddnode_ide1/Tensordot:output:0(node_ide1/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  @V
lambda/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Н
lambda/Gelu/mulMullambda/Gelu/mul/x:output:0node_ide1/BiasAdd:output:0*
T0*4
_output_shapes"
 :                  @W
lambda/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *є╡?Ц
lambda/Gelu/truedivRealDivnode_ide1/BiasAdd:output:0lambda/Gelu/Cast/x:output:0*
T0*4
_output_shapes"
 :                  @n
lambda/Gelu/ErfErflambda/Gelu/truediv:z:0*
T0*4
_output_shapes"
 :                  @V
lambda/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?И
lambda/Gelu/addAddV2lambda/Gelu/add/x:output:0lambda/Gelu/Erf:y:0*
T0*4
_output_shapes"
 :                  @Б
lambda/Gelu/mul_1Mullambda/Gelu/mul:z:0lambda/Gelu/add:z:0*
T0*4
_output_shapes"
 :                  @^
layer_normalization/ShapeShapelambda/Gelu/mul_1:z:0*
T0*
_output_shapes
:q
'layer_normalization/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)layer_normalization/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)layer_normalization/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╡
!layer_normalization/strided_sliceStridedSlice"layer_normalization/Shape:output:00layer_normalization/strided_slice/stack:output:02layer_normalization/strided_slice/stack_1:output:02layer_normalization/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
layer_normalization/mul/xConst*
_output_shapes
: *
dtype0*
value	B :П
layer_normalization/mulMul"layer_normalization/mul/x:output:0*layer_normalization/strided_slice:output:0*
T0*
_output_shapes
: s
)layer_normalization/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╜
#layer_normalization/strided_slice_1StridedSlice"layer_normalization/Shape:output:02layer_normalization/strided_slice_1/stack:output:04layer_normalization/strided_slice_1/stack_1:output:04layer_normalization/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskМ
layer_normalization/mul_1Mullayer_normalization/mul:z:0,layer_normalization/strided_slice_1:output:0*
T0*
_output_shapes
: s
)layer_normalization/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╜
#layer_normalization/strided_slice_2StridedSlice"layer_normalization/Shape:output:02layer_normalization/strided_slice_2/stack:output:04layer_normalization/strided_slice_2/stack_1:output:04layer_normalization/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :Х
layer_normalization/mul_2Mul$layer_normalization/mul_2/x:output:0,layer_normalization/strided_slice_2:output:0*
T0*
_output_shapes
: e
#layer_normalization/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :e
#layer_normalization/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :ё
!layer_normalization/Reshape/shapePack,layer_normalization/Reshape/shape/0:output:0layer_normalization/mul_1:z:0layer_normalization/mul_2:z:0,layer_normalization/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:г
layer_normalization/ReshapeReshapelambda/Gelu/mul_1:z:0*layer_normalization/Reshape/shape:output:0*
T0*/
_output_shapes
:         @t
layer_normalization/ones/packedPacklayer_normalization/mul_1:z:0*
N*
T0*
_output_shapes
:c
layer_normalization/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?б
layer_normalization/onesFill(layer_normalization/ones/packed:output:0'layer_normalization/ones/Const:output:0*
T0*#
_output_shapes
:         u
 layer_normalization/zeros/packedPacklayer_normalization/mul_1:z:0*
N*
T0*
_output_shapes
:d
layer_normalization/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    д
layer_normalization/zerosFill)layer_normalization/zeros/packed:output:0(layer_normalization/zeros/Const:output:0*
T0*#
_output_shapes
:         \
layer_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB ^
layer_normalization/Const_1Const*
_output_shapes
: *
dtype0*
valueB Ы
$layer_normalization/FusedBatchNormV3FusedBatchNormV3$layer_normalization/Reshape:output:0!layer_normalization/ones:output:0"layer_normalization/zeros:output:0"layer_normalization/Const:output:0$layer_normalization/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:         @:         :         :         :         :*
data_formatNCHW*
epsilon%oГ:╡
layer_normalization/Reshape_1Reshape(layer_normalization/FusedBatchNormV3:y:0"layer_normalization/Shape:output:0*
T0*4
_output_shapes"
 :                  @Ц
(layer_normalization/mul_3/ReadVariableOpReadVariableOp1layer_normalization_mul_3_readvariableop_resource*
_output_shapes
:@*
dtype0╣
layer_normalization/mul_3Mul&layer_normalization/Reshape_1:output:00layer_normalization/mul_3/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  @Т
&layer_normalization/add/ReadVariableOpReadVariableOp/layer_normalization_add_readvariableop_resource*
_output_shapes
:@*
dtype0о
layer_normalization/addAddV2layer_normalization/mul_3:z:0.layer_normalization/add/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  @О
"edge_ide1/Tensordot/ReadVariableOpReadVariableOp+edge_ide1_tensordot_readvariableop_resource*
_output_shapes

:@*
dtype0b
edge_ide1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:i
edge_ide1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       Q
edge_ide1/Tensordot/ShapeShapeinputs_1*
T0*
_output_shapes
:c
!edge_ide1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : у
edge_ide1/Tensordot/GatherV2GatherV2"edge_ide1/Tensordot/Shape:output:0!edge_ide1/Tensordot/free:output:0*edge_ide1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:e
#edge_ide1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ч
edge_ide1/Tensordot/GatherV2_1GatherV2"edge_ide1/Tensordot/Shape:output:0!edge_ide1/Tensordot/axes:output:0,edge_ide1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
edge_ide1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: М
edge_ide1/Tensordot/ProdProd%edge_ide1/Tensordot/GatherV2:output:0"edge_ide1/Tensordot/Const:output:0*
T0*
_output_shapes
: e
edge_ide1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Т
edge_ide1/Tensordot/Prod_1Prod'edge_ide1/Tensordot/GatherV2_1:output:0$edge_ide1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: a
edge_ide1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ─
edge_ide1/Tensordot/concatConcatV2!edge_ide1/Tensordot/free:output:0!edge_ide1/Tensordot/axes:output:0(edge_ide1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Ч
edge_ide1/Tensordot/stackPack!edge_ide1/Tensordot/Prod:output:0#edge_ide1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Ш
edge_ide1/Tensordot/transpose	Transposeinputs_1#edge_ide1/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :                  и
edge_ide1/Tensordot/ReshapeReshape!edge_ide1/Tensordot/transpose:y:0"edge_ide1/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  и
edge_ide1/Tensordot/MatMulMatMul$edge_ide1/Tensordot/Reshape:output:0*edge_ide1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @e
edge_ide1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@c
!edge_ide1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ╧
edge_ide1/Tensordot/concat_1ConcatV2%edge_ide1/Tensordot/GatherV2:output:0$edge_ide1/Tensordot/Const_2:output:0*edge_ide1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:к
edge_ide1/TensordotReshape$edge_ide1/Tensordot/MatMul:product:0%edge_ide1/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  @Ж
 edge_ide1/BiasAdd/ReadVariableOpReadVariableOp)edge_ide1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0г
edge_ide1/BiasAddBiasAddedge_ide1/Tensordot:output:0(edge_ide1/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  @О
"node_ide2/Tensordot/ReadVariableOpReadVariableOp+node_ide2_tensordot_readvariableop_resource*
_output_shapes

:@`*
dtype0b
node_ide2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:i
node_ide2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       d
node_ide2/Tensordot/ShapeShapelayer_normalization/add:z:0*
T0*
_output_shapes
:c
!node_ide2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : у
node_ide2/Tensordot/GatherV2GatherV2"node_ide2/Tensordot/Shape:output:0!node_ide2/Tensordot/free:output:0*node_ide2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:e
#node_ide2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ч
node_ide2/Tensordot/GatherV2_1GatherV2"node_ide2/Tensordot/Shape:output:0!node_ide2/Tensordot/axes:output:0,node_ide2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
node_ide2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: М
node_ide2/Tensordot/ProdProd%node_ide2/Tensordot/GatherV2:output:0"node_ide2/Tensordot/Const:output:0*
T0*
_output_shapes
: e
node_ide2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Т
node_ide2/Tensordot/Prod_1Prod'node_ide2/Tensordot/GatherV2_1:output:0$node_ide2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: a
node_ide2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ─
node_ide2/Tensordot/concatConcatV2!node_ide2/Tensordot/free:output:0!node_ide2/Tensordot/axes:output:0(node_ide2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Ч
node_ide2/Tensordot/stackPack!node_ide2/Tensordot/Prod:output:0#node_ide2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:л
node_ide2/Tensordot/transpose	Transposelayer_normalization/add:z:0#node_ide2/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :                  @и
node_ide2/Tensordot/ReshapeReshape!node_ide2/Tensordot/transpose:y:0"node_ide2/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  и
node_ide2/Tensordot/MatMulMatMul$node_ide2/Tensordot/Reshape:output:0*node_ide2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `e
node_ide2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:`c
!node_ide2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ╧
node_ide2/Tensordot/concat_1ConcatV2%node_ide2/Tensordot/GatherV2:output:0$node_ide2/Tensordot/Const_2:output:0*node_ide2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:к
node_ide2/TensordotReshape$node_ide2/Tensordot/MatMul:product:0%node_ide2/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  `Ж
 node_ide2/BiasAdd/ReadVariableOpReadVariableOp)node_ide2_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0г
node_ide2/BiasAddBiasAddnode_ide2/Tensordot:output:0(node_ide2/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `X
lambda/Gelu_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?С
lambda/Gelu_1/mulMullambda/Gelu_1/mul/x:output:0edge_ide1/BiasAdd:output:0*
T0*4
_output_shapes"
 :                  @Y
lambda/Gelu_1/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *є╡?Ъ
lambda/Gelu_1/truedivRealDivedge_ide1/BiasAdd:output:0lambda/Gelu_1/Cast/x:output:0*
T0*4
_output_shapes"
 :                  @r
lambda/Gelu_1/ErfErflambda/Gelu_1/truediv:z:0*
T0*4
_output_shapes"
 :                  @X
lambda/Gelu_1/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?О
lambda/Gelu_1/addAddV2lambda/Gelu_1/add/x:output:0lambda/Gelu_1/Erf:y:0*
T0*4
_output_shapes"
 :                  @З
lambda/Gelu_1/mul_1Mullambda/Gelu_1/mul:z:0lambda/Gelu_1/add:z:0*
T0*4
_output_shapes"
 :                  @X
lambda/Gelu_2/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?С
lambda/Gelu_2/mulMullambda/Gelu_2/mul/x:output:0node_ide2/BiasAdd:output:0*
T0*4
_output_shapes"
 :                  `Y
lambda/Gelu_2/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *є╡?Ъ
lambda/Gelu_2/truedivRealDivnode_ide2/BiasAdd:output:0lambda/Gelu_2/Cast/x:output:0*
T0*4
_output_shapes"
 :                  `r
lambda/Gelu_2/ErfErflambda/Gelu_2/truediv:z:0*
T0*4
_output_shapes"
 :                  `X
lambda/Gelu_2/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?О
lambda/Gelu_2/addAddV2lambda/Gelu_2/add/x:output:0lambda/Gelu_2/Erf:y:0*
T0*4
_output_shapes"
 :                  `З
lambda/Gelu_2/mul_1Mullambda/Gelu_2/mul:z:0lambda/Gelu_2/add:z:0*
T0*4
_output_shapes"
 :                  `b
layer_normalization_1/ShapeShapelambda/Gelu_1/mul_1:z:0*
T0*
_output_shapes
:s
)layer_normalization_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+layer_normalization_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┐
#layer_normalization_1/strided_sliceStridedSlice$layer_normalization_1/Shape:output:02layer_normalization_1/strided_slice/stack:output:04layer_normalization_1/strided_slice/stack_1:output:04layer_normalization_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization_1/mul/xConst*
_output_shapes
: *
dtype0*
value	B :Х
layer_normalization_1/mulMul$layer_normalization_1/mul/x:output:0,layer_normalization_1/strided_slice:output:0*
T0*
_output_shapes
: u
+layer_normalization_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╟
%layer_normalization_1/strided_slice_1StridedSlice$layer_normalization_1/Shape:output:04layer_normalization_1/strided_slice_1/stack:output:06layer_normalization_1/strided_slice_1/stack_1:output:06layer_normalization_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskТ
layer_normalization_1/mul_1Mullayer_normalization_1/mul:z:0.layer_normalization_1/strided_slice_1:output:0*
T0*
_output_shapes
: u
+layer_normalization_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╟
%layer_normalization_1/strided_slice_2StridedSlice$layer_normalization_1/Shape:output:04layer_normalization_1/strided_slice_2/stack:output:06layer_normalization_1/strided_slice_2/stack_1:output:06layer_normalization_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
layer_normalization_1/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :Ы
layer_normalization_1/mul_2Mul&layer_normalization_1/mul_2/x:output:0.layer_normalization_1/strided_slice_2:output:0*
T0*
_output_shapes
: g
%layer_normalization_1/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :g
%layer_normalization_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :√
#layer_normalization_1/Reshape/shapePack.layer_normalization_1/Reshape/shape/0:output:0layer_normalization_1/mul_1:z:0layer_normalization_1/mul_2:z:0.layer_normalization_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:й
layer_normalization_1/ReshapeReshapelambda/Gelu_1/mul_1:z:0,layer_normalization_1/Reshape/shape:output:0*
T0*/
_output_shapes
:         @x
!layer_normalization_1/ones/packedPacklayer_normalization_1/mul_1:z:0*
N*
T0*
_output_shapes
:e
 layer_normalization_1/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?з
layer_normalization_1/onesFill*layer_normalization_1/ones/packed:output:0)layer_normalization_1/ones/Const:output:0*
T0*#
_output_shapes
:         y
"layer_normalization_1/zeros/packedPacklayer_normalization_1/mul_1:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    к
layer_normalization_1/zerosFill+layer_normalization_1/zeros/packed:output:0*layer_normalization_1/zeros/Const:output:0*
T0*#
_output_shapes
:         ^
layer_normalization_1/ConstConst*
_output_shapes
: *
dtype0*
valueB `
layer_normalization_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB з
&layer_normalization_1/FusedBatchNormV3FusedBatchNormV3&layer_normalization_1/Reshape:output:0#layer_normalization_1/ones:output:0$layer_normalization_1/zeros:output:0$layer_normalization_1/Const:output:0&layer_normalization_1/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:         @:         :         :         :         :*
data_formatNCHW*
epsilon%oГ:╗
layer_normalization_1/Reshape_1Reshape*layer_normalization_1/FusedBatchNormV3:y:0$layer_normalization_1/Shape:output:0*
T0*4
_output_shapes"
 :                  @Ъ
*layer_normalization_1/mul_3/ReadVariableOpReadVariableOp3layer_normalization_1_mul_3_readvariableop_resource*
_output_shapes
:@*
dtype0┐
layer_normalization_1/mul_3Mul(layer_normalization_1/Reshape_1:output:02layer_normalization_1/mul_3/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  @Ц
(layer_normalization_1/add/ReadVariableOpReadVariableOp1layer_normalization_1_add_readvariableop_resource*
_output_shapes
:@*
dtype0┤
layer_normalization_1/addAddV2layer_normalization_1/mul_3:z:00layer_normalization_1/add/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  @О
"edge_ide2/Tensordot/ReadVariableOpReadVariableOp+edge_ide2_tensordot_readvariableop_resource*
_output_shapes

:@`*
dtype0b
edge_ide2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:i
edge_ide2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       f
edge_ide2/Tensordot/ShapeShapelayer_normalization_1/add:z:0*
T0*
_output_shapes
:c
!edge_ide2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : у
edge_ide2/Tensordot/GatherV2GatherV2"edge_ide2/Tensordot/Shape:output:0!edge_ide2/Tensordot/free:output:0*edge_ide2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:e
#edge_ide2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ч
edge_ide2/Tensordot/GatherV2_1GatherV2"edge_ide2/Tensordot/Shape:output:0!edge_ide2/Tensordot/axes:output:0,edge_ide2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
edge_ide2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: М
edge_ide2/Tensordot/ProdProd%edge_ide2/Tensordot/GatherV2:output:0"edge_ide2/Tensordot/Const:output:0*
T0*
_output_shapes
: e
edge_ide2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Т
edge_ide2/Tensordot/Prod_1Prod'edge_ide2/Tensordot/GatherV2_1:output:0$edge_ide2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: a
edge_ide2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ─
edge_ide2/Tensordot/concatConcatV2!edge_ide2/Tensordot/free:output:0!edge_ide2/Tensordot/axes:output:0(edge_ide2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Ч
edge_ide2/Tensordot/stackPack!edge_ide2/Tensordot/Prod:output:0#edge_ide2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:н
edge_ide2/Tensordot/transpose	Transposelayer_normalization_1/add:z:0#edge_ide2/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :                  @и
edge_ide2/Tensordot/ReshapeReshape!edge_ide2/Tensordot/transpose:y:0"edge_ide2/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  и
edge_ide2/Tensordot/MatMulMatMul$edge_ide2/Tensordot/Reshape:output:0*edge_ide2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `e
edge_ide2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:`c
!edge_ide2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ╧
edge_ide2/Tensordot/concat_1ConcatV2%edge_ide2/Tensordot/GatherV2:output:0$edge_ide2/Tensordot/Const_2:output:0*edge_ide2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:к
edge_ide2/TensordotReshape$edge_ide2/Tensordot/MatMul:product:0%edge_ide2/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  `Ж
 edge_ide2/BiasAdd/ReadVariableOpReadVariableOp)edge_ide2_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0г
edge_ide2/BiasAddBiasAddedge_ide2/Tensordot:output:0(edge_ide2/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `}
,tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
.tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
.tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╩
&tf.__operators__.getitem/strided_sliceStridedSliceinputs_15tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*4
_output_shapes"
 :                  *
ellipsis_maskX
lambda/Gelu_3/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?С
lambda/Gelu_3/mulMullambda/Gelu_3/mul/x:output:0edge_ide2/BiasAdd:output:0*
T0*4
_output_shapes"
 :                  `Y
lambda/Gelu_3/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *є╡?Ъ
lambda/Gelu_3/truedivRealDivedge_ide2/BiasAdd:output:0lambda/Gelu_3/Cast/x:output:0*
T0*4
_output_shapes"
 :                  `r
lambda/Gelu_3/ErfErflambda/Gelu_3/truediv:z:0*
T0*4
_output_shapes"
 :                  `X
lambda/Gelu_3/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?О
lambda/Gelu_3/addAddV2lambda/Gelu_3/add/x:output:0lambda/Gelu_3/Erf:y:0*
T0*4
_output_shapes"
 :                  `З
lambda/Gelu_3/mul_1Mullambda/Gelu_3/mul:z:0lambda/Gelu_3/add:z:0*
T0*4
_output_shapes"
 :                  `b
layer_normalization_2/ShapeShapelambda/Gelu_2/mul_1:z:0*
T0*
_output_shapes
:s
)layer_normalization_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+layer_normalization_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┐
#layer_normalization_2/strided_sliceStridedSlice$layer_normalization_2/Shape:output:02layer_normalization_2/strided_slice/stack:output:04layer_normalization_2/strided_slice/stack_1:output:04layer_normalization_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization_2/mul/xConst*
_output_shapes
: *
dtype0*
value	B :Х
layer_normalization_2/mulMul$layer_normalization_2/mul/x:output:0,layer_normalization_2/strided_slice:output:0*
T0*
_output_shapes
: u
+layer_normalization_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╟
%layer_normalization_2/strided_slice_1StridedSlice$layer_normalization_2/Shape:output:04layer_normalization_2/strided_slice_1/stack:output:06layer_normalization_2/strided_slice_1/stack_1:output:06layer_normalization_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskТ
layer_normalization_2/mul_1Mullayer_normalization_2/mul:z:0.layer_normalization_2/strided_slice_1:output:0*
T0*
_output_shapes
: u
+layer_normalization_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╟
%layer_normalization_2/strided_slice_2StridedSlice$layer_normalization_2/Shape:output:04layer_normalization_2/strided_slice_2/stack:output:06layer_normalization_2/strided_slice_2/stack_1:output:06layer_normalization_2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
layer_normalization_2/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :Ы
layer_normalization_2/mul_2Mul&layer_normalization_2/mul_2/x:output:0.layer_normalization_2/strided_slice_2:output:0*
T0*
_output_shapes
: g
%layer_normalization_2/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :g
%layer_normalization_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :√
#layer_normalization_2/Reshape/shapePack.layer_normalization_2/Reshape/shape/0:output:0layer_normalization_2/mul_1:z:0layer_normalization_2/mul_2:z:0.layer_normalization_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:й
layer_normalization_2/ReshapeReshapelambda/Gelu_2/mul_1:z:0,layer_normalization_2/Reshape/shape:output:0*
T0*/
_output_shapes
:         `x
!layer_normalization_2/ones/packedPacklayer_normalization_2/mul_1:z:0*
N*
T0*
_output_shapes
:e
 layer_normalization_2/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?з
layer_normalization_2/onesFill*layer_normalization_2/ones/packed:output:0)layer_normalization_2/ones/Const:output:0*
T0*#
_output_shapes
:         y
"layer_normalization_2/zeros/packedPacklayer_normalization_2/mul_1:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    к
layer_normalization_2/zerosFill+layer_normalization_2/zeros/packed:output:0*layer_normalization_2/zeros/Const:output:0*
T0*#
_output_shapes
:         ^
layer_normalization_2/ConstConst*
_output_shapes
: *
dtype0*
valueB `
layer_normalization_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB з
&layer_normalization_2/FusedBatchNormV3FusedBatchNormV3&layer_normalization_2/Reshape:output:0#layer_normalization_2/ones:output:0$layer_normalization_2/zeros:output:0$layer_normalization_2/Const:output:0&layer_normalization_2/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:         `:         :         :         :         :*
data_formatNCHW*
epsilon%oГ:╗
layer_normalization_2/Reshape_1Reshape*layer_normalization_2/FusedBatchNormV3:y:0$layer_normalization_2/Shape:output:0*
T0*4
_output_shapes"
 :                  `Ъ
*layer_normalization_2/mul_3/ReadVariableOpReadVariableOp3layer_normalization_2_mul_3_readvariableop_resource*
_output_shapes
:`*
dtype0┐
layer_normalization_2/mul_3Mul(layer_normalization_2/Reshape_1:output:02layer_normalization_2/mul_3/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `Ц
(layer_normalization_2/add/ReadVariableOpReadVariableOp1layer_normalization_2_add_readvariableop_resource*
_output_shapes
:`*
dtype0┤
layer_normalization_2/addAddV2layer_normalization_2/mul_3:z:00layer_normalization_2/add/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `b
layer_normalization_3/ShapeShapelambda/Gelu_3/mul_1:z:0*
T0*
_output_shapes
:s
)layer_normalization_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+layer_normalization_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┐
#layer_normalization_3/strided_sliceStridedSlice$layer_normalization_3/Shape:output:02layer_normalization_3/strided_slice/stack:output:04layer_normalization_3/strided_slice/stack_1:output:04layer_normalization_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization_3/mul/xConst*
_output_shapes
: *
dtype0*
value	B :Х
layer_normalization_3/mulMul$layer_normalization_3/mul/x:output:0,layer_normalization_3/strided_slice:output:0*
T0*
_output_shapes
: u
+layer_normalization_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╟
%layer_normalization_3/strided_slice_1StridedSlice$layer_normalization_3/Shape:output:04layer_normalization_3/strided_slice_1/stack:output:06layer_normalization_3/strided_slice_1/stack_1:output:06layer_normalization_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskТ
layer_normalization_3/mul_1Mullayer_normalization_3/mul:z:0.layer_normalization_3/strided_slice_1:output:0*
T0*
_output_shapes
: u
+layer_normalization_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╟
%layer_normalization_3/strided_slice_2StridedSlice$layer_normalization_3/Shape:output:04layer_normalization_3/strided_slice_2/stack:output:06layer_normalization_3/strided_slice_2/stack_1:output:06layer_normalization_3/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
layer_normalization_3/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :Ы
layer_normalization_3/mul_2Mul&layer_normalization_3/mul_2/x:output:0.layer_normalization_3/strided_slice_2:output:0*
T0*
_output_shapes
: g
%layer_normalization_3/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :g
%layer_normalization_3/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :√
#layer_normalization_3/Reshape/shapePack.layer_normalization_3/Reshape/shape/0:output:0layer_normalization_3/mul_1:z:0layer_normalization_3/mul_2:z:0.layer_normalization_3/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:й
layer_normalization_3/ReshapeReshapelambda/Gelu_3/mul_1:z:0,layer_normalization_3/Reshape/shape:output:0*
T0*/
_output_shapes
:         `x
!layer_normalization_3/ones/packedPacklayer_normalization_3/mul_1:z:0*
N*
T0*
_output_shapes
:e
 layer_normalization_3/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?з
layer_normalization_3/onesFill*layer_normalization_3/ones/packed:output:0)layer_normalization_3/ones/Const:output:0*
T0*#
_output_shapes
:         y
"layer_normalization_3/zeros/packedPacklayer_normalization_3/mul_1:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization_3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    к
layer_normalization_3/zerosFill+layer_normalization_3/zeros/packed:output:0*layer_normalization_3/zeros/Const:output:0*
T0*#
_output_shapes
:         ^
layer_normalization_3/ConstConst*
_output_shapes
: *
dtype0*
valueB `
layer_normalization_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB з
&layer_normalization_3/FusedBatchNormV3FusedBatchNormV3&layer_normalization_3/Reshape:output:0#layer_normalization_3/ones:output:0$layer_normalization_3/zeros:output:0$layer_normalization_3/Const:output:0&layer_normalization_3/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:         `:         :         :         :         :*
data_formatNCHW*
epsilon%oГ:╗
layer_normalization_3/Reshape_1Reshape*layer_normalization_3/FusedBatchNormV3:y:0$layer_normalization_3/Shape:output:0*
T0*4
_output_shapes"
 :                  `Ъ
*layer_normalization_3/mul_3/ReadVariableOpReadVariableOp3layer_normalization_3_mul_3_readvariableop_resource*
_output_shapes
:`*
dtype0┐
layer_normalization_3/mul_3Mul(layer_normalization_3/Reshape_1:output:02layer_normalization_3/mul_3/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `Ц
(layer_normalization_3/add/ReadVariableOpReadVariableOp1layer_normalization_3_add_readvariableop_resource*
_output_shapes
:`*
dtype0┤
layer_normalization_3/addAddV2layer_normalization_3/mul_3:z:00layer_normalization_3/add/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `{
tf.ones_like/ones_like/ShapeShape/tf.__operators__.getitem/strided_slice:output:0*
T0*
_output_shapes
:a
tf.ones_like/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?л
tf.ones_like/ones_likeFill%tf.ones_like/ones_like/Shape:output:0%tf.ones_like/ones_like/Const:output:0*
T0*4
_output_shapes"
 :                  V
	mpn/ShapeShapelayer_normalization_2/add:z:0*
T0*
_output_shapes
:a
mpn/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:c
mpn/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
mpn/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
mpn/strided_sliceStridedSlicempn/Shape:output:0 mpn/strided_slice/stack:output:0"mpn/strided_slice/stack_1:output:0"mpn/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskC
mpn/Shape_1Shapeinputs_2*
T0*
_output_shapes
:c
mpn/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:e
mpn/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
mpn/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:я
mpn/strided_slice_1StridedSlicempn/Shape_1:output:0"mpn/strided_slice_1/stack:output:0$mpn/strided_slice_1/stack_1:output:0$mpn/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
mpn/Shape_2Shapelayer_normalization_2/add:z:0*
T0*
_output_shapes
:c
mpn/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: e
mpn/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
mpn/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:я
mpn/strided_slice_2StridedSlicempn/Shape_2:output:0"mpn/strided_slice_2/stack:output:0$mpn/strided_slice_2/stack_1:output:0$mpn/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskS
mpn/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :╒
mpn/GatherV2GatherV2layer_normalization_2/add:z:0inputs_2mpn/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*8
_output_shapes&
$:"                  `*

batch_dimsV
mpn/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :└б
mpn/Reshape/shapePackmpn/strided_slice_2:output:0mpn/strided_slice_1:output:0mpn/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:Й
mpn/ReshapeReshapempn/GatherV2:output:0mpn/Reshape/shape:output:0*
T0*5
_output_shapes#
!:                  └Z
mpn/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
         о

mpn/concatConcatV2mpn/Reshape:output:0layer_normalization_3/add:z:0mpn/concat/axis:output:0*
N*
T0*5
_output_shapes#
!:                  ае
-mpn/sequential/dense/Tensordot/ReadVariableOpReadVariableOp6mpn_sequential_dense_tensordot_readvariableop_resource*
_output_shapes
:	а`*
dtype0m
#mpn/sequential/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:t
#mpn/sequential/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       g
$mpn/sequential/dense/Tensordot/ShapeShapempn/concat:output:0*
T0*
_output_shapes
:n
,mpn/sequential/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : П
'mpn/sequential/dense/Tensordot/GatherV2GatherV2-mpn/sequential/dense/Tensordot/Shape:output:0,mpn/sequential/dense/Tensordot/free:output:05mpn/sequential/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:p
.mpn/sequential/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : У
)mpn/sequential/dense/Tensordot/GatherV2_1GatherV2-mpn/sequential/dense/Tensordot/Shape:output:0,mpn/sequential/dense/Tensordot/axes:output:07mpn/sequential/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
$mpn/sequential/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: н
#mpn/sequential/dense/Tensordot/ProdProd0mpn/sequential/dense/Tensordot/GatherV2:output:0-mpn/sequential/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: p
&mpn/sequential/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: │
%mpn/sequential/dense/Tensordot/Prod_1Prod2mpn/sequential/dense/Tensordot/GatherV2_1:output:0/mpn/sequential/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: l
*mpn/sequential/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ё
%mpn/sequential/dense/Tensordot/concatConcatV2,mpn/sequential/dense/Tensordot/free:output:0,mpn/sequential/dense/Tensordot/axes:output:03mpn/sequential/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:╕
$mpn/sequential/dense/Tensordot/stackPack,mpn/sequential/dense/Tensordot/Prod:output:0.mpn/sequential/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:║
(mpn/sequential/dense/Tensordot/transpose	Transposempn/concat:output:0.mpn/sequential/dense/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:                  а╔
&mpn/sequential/dense/Tensordot/ReshapeReshape,mpn/sequential/dense/Tensordot/transpose:y:0-mpn/sequential/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  ╔
%mpn/sequential/dense/Tensordot/MatMulMatMul/mpn/sequential/dense/Tensordot/Reshape:output:05mpn/sequential/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `p
&mpn/sequential/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:`n
,mpn/sequential/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : √
'mpn/sequential/dense/Tensordot/concat_1ConcatV20mpn/sequential/dense/Tensordot/GatherV2:output:0/mpn/sequential/dense/Tensordot/Const_2:output:05mpn/sequential/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:╦
mpn/sequential/dense/TensordotReshape/mpn/sequential/dense/Tensordot/MatMul:product:00mpn/sequential/dense/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  `Ь
+mpn/sequential/dense/BiasAdd/ReadVariableOpReadVariableOp4mpn_sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0─
mpn/sequential/dense/BiasAddBiasAdd'mpn/sequential/dense/Tensordot:output:03mpn/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `e
 mpn/sequential/lambda/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?╢
mpn/sequential/lambda/Gelu/mulMul)mpn/sequential/lambda/Gelu/mul/x:output:0%mpn/sequential/dense/BiasAdd:output:0*
T0*4
_output_shapes"
 :                  `f
!mpn/sequential/lambda/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *є╡?┐
"mpn/sequential/lambda/Gelu/truedivRealDiv%mpn/sequential/dense/BiasAdd:output:0*mpn/sequential/lambda/Gelu/Cast/x:output:0*
T0*4
_output_shapes"
 :                  `М
mpn/sequential/lambda/Gelu/ErfErf&mpn/sequential/lambda/Gelu/truediv:z:0*
T0*4
_output_shapes"
 :                  `e
 mpn/sequential/lambda/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╡
mpn/sequential/lambda/Gelu/addAddV2)mpn/sequential/lambda/Gelu/add/x:output:0"mpn/sequential/lambda/Gelu/Erf:y:0*
T0*4
_output_shapes"
 :                  `о
 mpn/sequential/lambda/Gelu/mul_1Mul"mpn/sequential/lambda/Gelu/mul:z:0"mpn/sequential/lambda/Gelu/add:z:0*
T0*4
_output_shapes"
 :                  `~
*mpn/sequential/layer_normalization_4/ShapeShape$mpn/sequential/lambda/Gelu/mul_1:z:0*
T0*
_output_shapes
:В
8mpn/sequential/layer_normalization_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Д
:mpn/sequential/layer_normalization_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Д
:mpn/sequential/layer_normalization_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:К
2mpn/sequential/layer_normalization_4/strided_sliceStridedSlice3mpn/sequential/layer_normalization_4/Shape:output:0Ampn/sequential/layer_normalization_4/strided_slice/stack:output:0Cmpn/sequential/layer_normalization_4/strided_slice/stack_1:output:0Cmpn/sequential/layer_normalization_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
*mpn/sequential/layer_normalization_4/mul/xConst*
_output_shapes
: *
dtype0*
value	B :┬
(mpn/sequential/layer_normalization_4/mulMul3mpn/sequential/layer_normalization_4/mul/x:output:0;mpn/sequential/layer_normalization_4/strided_slice:output:0*
T0*
_output_shapes
: Д
:mpn/sequential/layer_normalization_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:Ж
<mpn/sequential/layer_normalization_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ж
<mpn/sequential/layer_normalization_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Т
4mpn/sequential/layer_normalization_4/strided_slice_1StridedSlice3mpn/sequential/layer_normalization_4/Shape:output:0Cmpn/sequential/layer_normalization_4/strided_slice_1/stack:output:0Empn/sequential/layer_normalization_4/strided_slice_1/stack_1:output:0Empn/sequential/layer_normalization_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask┐
*mpn/sequential/layer_normalization_4/mul_1Mul,mpn/sequential/layer_normalization_4/mul:z:0=mpn/sequential/layer_normalization_4/strided_slice_1:output:0*
T0*
_output_shapes
: Д
:mpn/sequential/layer_normalization_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:Ж
<mpn/sequential/layer_normalization_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ж
<mpn/sequential/layer_normalization_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Т
4mpn/sequential/layer_normalization_4/strided_slice_2StridedSlice3mpn/sequential/layer_normalization_4/Shape:output:0Cmpn/sequential/layer_normalization_4/strided_slice_2/stack:output:0Empn/sequential/layer_normalization_4/strided_slice_2/stack_1:output:0Empn/sequential/layer_normalization_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
,mpn/sequential/layer_normalization_4/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :╚
*mpn/sequential/layer_normalization_4/mul_2Mul5mpn/sequential/layer_normalization_4/mul_2/x:output:0=mpn/sequential/layer_normalization_4/strided_slice_2:output:0*
T0*
_output_shapes
: v
4mpn/sequential/layer_normalization_4/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :v
4mpn/sequential/layer_normalization_4/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :╞
2mpn/sequential/layer_normalization_4/Reshape/shapePack=mpn/sequential/layer_normalization_4/Reshape/shape/0:output:0.mpn/sequential/layer_normalization_4/mul_1:z:0.mpn/sequential/layer_normalization_4/mul_2:z:0=mpn/sequential/layer_normalization_4/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:╘
,mpn/sequential/layer_normalization_4/ReshapeReshape$mpn/sequential/lambda/Gelu/mul_1:z:0;mpn/sequential/layer_normalization_4/Reshape/shape:output:0*
T0*/
_output_shapes
:         `Ц
0mpn/sequential/layer_normalization_4/ones/packedPack.mpn/sequential/layer_normalization_4/mul_1:z:0*
N*
T0*
_output_shapes
:t
/mpn/sequential/layer_normalization_4/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╘
)mpn/sequential/layer_normalization_4/onesFill9mpn/sequential/layer_normalization_4/ones/packed:output:08mpn/sequential/layer_normalization_4/ones/Const:output:0*
T0*#
_output_shapes
:         Ч
1mpn/sequential/layer_normalization_4/zeros/packedPack.mpn/sequential/layer_normalization_4/mul_1:z:0*
N*
T0*
_output_shapes
:u
0mpn/sequential/layer_normalization_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ╫
*mpn/sequential/layer_normalization_4/zerosFill:mpn/sequential/layer_normalization_4/zeros/packed:output:09mpn/sequential/layer_normalization_4/zeros/Const:output:0*
T0*#
_output_shapes
:         m
*mpn/sequential/layer_normalization_4/ConstConst*
_output_shapes
: *
dtype0*
valueB o
,mpn/sequential/layer_normalization_4/Const_1Const*
_output_shapes
: *
dtype0*
valueB Б
5mpn/sequential/layer_normalization_4/FusedBatchNormV3FusedBatchNormV35mpn/sequential/layer_normalization_4/Reshape:output:02mpn/sequential/layer_normalization_4/ones:output:03mpn/sequential/layer_normalization_4/zeros:output:03mpn/sequential/layer_normalization_4/Const:output:05mpn/sequential/layer_normalization_4/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:         `:         :         :         :         :*
data_formatNCHW*
epsilon%oГ:ш
.mpn/sequential/layer_normalization_4/Reshape_1Reshape9mpn/sequential/layer_normalization_4/FusedBatchNormV3:y:03mpn/sequential/layer_normalization_4/Shape:output:0*
T0*4
_output_shapes"
 :                  `╕
9mpn/sequential/layer_normalization_4/mul_3/ReadVariableOpReadVariableOpBmpn_sequential_layer_normalization_4_mul_3_readvariableop_resource*
_output_shapes
:`*
dtype0ь
*mpn/sequential/layer_normalization_4/mul_3Mul7mpn/sequential/layer_normalization_4/Reshape_1:output:0Ampn/sequential/layer_normalization_4/mul_3/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `┤
7mpn/sequential/layer_normalization_4/add/ReadVariableOpReadVariableOp@mpn_sequential_layer_normalization_4_add_readvariableop_resource*
_output_shapes
:`*
dtype0с
(mpn/sequential/layer_normalization_4/addAddV2.mpn/sequential/layer_normalization_4/mul_3:z:0?mpn/sequential/layer_normalization_4/add/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `Ь
mpn/mulMul,mpn/sequential/layer_normalization_4/add:z:0tf.ones_like/ones_like:output:0*
T0*4
_output_shapes"
 :                  `T
mpn/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :`
mpn/zeros/packedPackmpn/strided_slice:output:0mpn/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:T
mpn/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    x
	mpn/zerosFillmpn/zeros/packed:output:0mpn/zeros/Const:output:0*
T0*'
_output_shapes
:         `I
mpn/scan/ShapeShapempn/mul:z:0*
T0*
_output_shapes
:f
mpn/scan/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
mpn/scan/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
mpn/scan/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:■
mpn/scan/strided_sliceStridedSlicempn/scan/Shape:output:0%mpn/scan/strided_slice/stack:output:0'mpn/scan/strided_slice/stack_1:output:0'mpn/scan/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
$mpn/scan/TensorArrayV2/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    `   ═
mpn/scan/TensorArrayV2TensorListReserve-mpn/scan/TensorArrayV2/element_shape:output:0mpn/scan/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥w
&mpn/scan/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╤
mpn/scan/TensorArrayV2_1TensorListReserve/mpn/scan/TensorArrayV2_1/element_shape:output:0mpn/scan/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╧w
&mpn/scan/TensorArrayV2_2/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╤
mpn/scan/TensorArrayV2_2TensorListReserve/mpn/scan/TensorArrayV2_2/element_shape:output:0mpn/scan/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥П
>mpn/scan/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    `   Ё
0mpn/scan/TensorArrayUnstack/TensorListFromTensorTensorListFromTensormpn/mul:z:0Gmpn/scan/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥С
@mpn/scan/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ё
2mpn/scan/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensorinputs_2Impn/scan/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╧С
@mpn/scan/TensorArrayUnstack_2/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ё
2mpn/scan/TensorArrayUnstack_2/TensorListFromTensorTensorListFromTensorinputs_3Impn/scan/TensorArrayUnstack_2/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥w
&mpn/scan/TensorArrayV2_3/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    `   ╤
mpn/scan/TensorArrayV2_3TensorListReserve/mpn/scan/TensorArrayV2_3/element_shape:output:0mpn/scan/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥P
mpn/scan/ConstConst*
_output_shapes
: *
dtype0*
value	B : ]
mpn/scan/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : У
mpn/scan/whileStatelessWhile$mpn/scan/while/loop_counter:output:0mpn/scan/strided_slice:output:0mpn/scan/Const:output:0mpn/zeros:output:0!mpn/scan/TensorArrayV2_3:handle:0mpn/scan/strided_slice:output:0@mpn/scan/TensorArrayUnstack/TensorListFromTensor:output_handle:0Bmpn/scan/TensorArrayUnstack_1/TensorListFromTensor:output_handle:0Bmpn/scan/TensorArrayUnstack_2/TensorListFromTensor:output_handle:0mpn/strided_slice:output:0*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : :         `: : : : : : * 
_read_only_resource_inputs
 *
_stateful_parallelism( *%
bodyR
mpn_scan_while_body_18595*%
condR
mpn_scan_while_cond_18594*8
output_shapes'
%: : : :         `: : : : : : К
9mpn/scan/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    `   ц
+mpn/scan/TensorArrayV2Stack/TensorListStackTensorListStackmpn/scan/while:output:4Bmpn/scan/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  `*
element_dtype0c
mpn/lambda_1/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
         р
mpn/lambda_1/concatConcatV2layer_normalization_2/add:z:04mpn/scan/TensorArrayV2Stack/TensorListStack:tensor:0!mpn/lambda_1/concat/axis:output:0*
N*
T0*5
_output_shapes#
!:                  └У
$mpn/dense_1/Tensordot/ReadVariableOpReadVariableOp-mpn_dense_1_tensordot_readvariableop_resource*
_output_shapes
:	└`*
dtype0d
mpn/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:k
mpn/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       g
mpn/dense_1/Tensordot/ShapeShapempn/lambda_1/concat:output:0*
T0*
_output_shapes
:e
#mpn/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ы
mpn/dense_1/Tensordot/GatherV2GatherV2$mpn/dense_1/Tensordot/Shape:output:0#mpn/dense_1/Tensordot/free:output:0,mpn/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:g
%mpn/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : я
 mpn/dense_1/Tensordot/GatherV2_1GatherV2$mpn/dense_1/Tensordot/Shape:output:0#mpn/dense_1/Tensordot/axes:output:0.mpn/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:e
mpn/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: Т
mpn/dense_1/Tensordot/ProdProd'mpn/dense_1/Tensordot/GatherV2:output:0$mpn/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: g
mpn/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Ш
mpn/dense_1/Tensordot/Prod_1Prod)mpn/dense_1/Tensordot/GatherV2_1:output:0&mpn/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: c
!mpn/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ╠
mpn/dense_1/Tensordot/concatConcatV2#mpn/dense_1/Tensordot/free:output:0#mpn/dense_1/Tensordot/axes:output:0*mpn/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Э
mpn/dense_1/Tensordot/stackPack#mpn/dense_1/Tensordot/Prod:output:0%mpn/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:▒
mpn/dense_1/Tensordot/transpose	Transposempn/lambda_1/concat:output:0%mpn/dense_1/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:                  └о
mpn/dense_1/Tensordot/ReshapeReshape#mpn/dense_1/Tensordot/transpose:y:0$mpn/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  о
mpn/dense_1/Tensordot/MatMulMatMul&mpn/dense_1/Tensordot/Reshape:output:0,mpn/dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `g
mpn/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:`e
#mpn/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ╫
mpn/dense_1/Tensordot/concat_1ConcatV2'mpn/dense_1/Tensordot/GatherV2:output:0&mpn/dense_1/Tensordot/Const_2:output:0,mpn/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:░
mpn/dense_1/TensordotReshape&mpn/dense_1/Tensordot/MatMul:product:0'mpn/dense_1/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  `К
"mpn/dense_1/BiasAdd/ReadVariableOpReadVariableOp+mpn_dense_1_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0й
mpn/dense_1/BiasAddBiasAddmpn/dense_1/Tensordot:output:0*mpn/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `g
"mpn/sequential_1/lambda/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?▒
 mpn/sequential_1/lambda/Gelu/mulMul+mpn/sequential_1/lambda/Gelu/mul/x:output:0mpn/dense_1/BiasAdd:output:0*
T0*4
_output_shapes"
 :                  `h
#mpn/sequential_1/lambda/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *є╡?║
$mpn/sequential_1/lambda/Gelu/truedivRealDivmpn/dense_1/BiasAdd:output:0,mpn/sequential_1/lambda/Gelu/Cast/x:output:0*
T0*4
_output_shapes"
 :                  `Р
 mpn/sequential_1/lambda/Gelu/ErfErf(mpn/sequential_1/lambda/Gelu/truediv:z:0*
T0*4
_output_shapes"
 :                  `g
"mpn/sequential_1/lambda/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╗
 mpn/sequential_1/lambda/Gelu/addAddV2+mpn/sequential_1/lambda/Gelu/add/x:output:0$mpn/sequential_1/lambda/Gelu/Erf:y:0*
T0*4
_output_shapes"
 :                  `┤
"mpn/sequential_1/lambda/Gelu/mul_1Mul$mpn/sequential_1/lambda/Gelu/mul:z:0$mpn/sequential_1/lambda/Gelu/add:z:0*
T0*4
_output_shapes"
 :                  `В
,mpn/sequential_1/layer_normalization_5/ShapeShape&mpn/sequential_1/lambda/Gelu/mul_1:z:0*
T0*
_output_shapes
:Д
:mpn/sequential_1/layer_normalization_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Ж
<mpn/sequential_1/layer_normalization_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ж
<mpn/sequential_1/layer_normalization_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ф
4mpn/sequential_1/layer_normalization_5/strided_sliceStridedSlice5mpn/sequential_1/layer_normalization_5/Shape:output:0Cmpn/sequential_1/layer_normalization_5/strided_slice/stack:output:0Empn/sequential_1/layer_normalization_5/strided_slice/stack_1:output:0Empn/sequential_1/layer_normalization_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
,mpn/sequential_1/layer_normalization_5/mul/xConst*
_output_shapes
: *
dtype0*
value	B :╚
*mpn/sequential_1/layer_normalization_5/mulMul5mpn/sequential_1/layer_normalization_5/mul/x:output:0=mpn/sequential_1/layer_normalization_5/strided_slice:output:0*
T0*
_output_shapes
: Ж
<mpn/sequential_1/layer_normalization_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:И
>mpn/sequential_1/layer_normalization_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:И
>mpn/sequential_1/layer_normalization_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ь
6mpn/sequential_1/layer_normalization_5/strided_slice_1StridedSlice5mpn/sequential_1/layer_normalization_5/Shape:output:0Empn/sequential_1/layer_normalization_5/strided_slice_1/stack:output:0Gmpn/sequential_1/layer_normalization_5/strided_slice_1/stack_1:output:0Gmpn/sequential_1/layer_normalization_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask┼
,mpn/sequential_1/layer_normalization_5/mul_1Mul.mpn/sequential_1/layer_normalization_5/mul:z:0?mpn/sequential_1/layer_normalization_5/strided_slice_1:output:0*
T0*
_output_shapes
: Ж
<mpn/sequential_1/layer_normalization_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:И
>mpn/sequential_1/layer_normalization_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:И
>mpn/sequential_1/layer_normalization_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ь
6mpn/sequential_1/layer_normalization_5/strided_slice_2StridedSlice5mpn/sequential_1/layer_normalization_5/Shape:output:0Empn/sequential_1/layer_normalization_5/strided_slice_2/stack:output:0Gmpn/sequential_1/layer_normalization_5/strided_slice_2/stack_1:output:0Gmpn/sequential_1/layer_normalization_5/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskp
.mpn/sequential_1/layer_normalization_5/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :╬
,mpn/sequential_1/layer_normalization_5/mul_2Mul7mpn/sequential_1/layer_normalization_5/mul_2/x:output:0?mpn/sequential_1/layer_normalization_5/strided_slice_2:output:0*
T0*
_output_shapes
: x
6mpn/sequential_1/layer_normalization_5/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :x
6mpn/sequential_1/layer_normalization_5/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :╨
4mpn/sequential_1/layer_normalization_5/Reshape/shapePack?mpn/sequential_1/layer_normalization_5/Reshape/shape/0:output:00mpn/sequential_1/layer_normalization_5/mul_1:z:00mpn/sequential_1/layer_normalization_5/mul_2:z:0?mpn/sequential_1/layer_normalization_5/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:┌
.mpn/sequential_1/layer_normalization_5/ReshapeReshape&mpn/sequential_1/lambda/Gelu/mul_1:z:0=mpn/sequential_1/layer_normalization_5/Reshape/shape:output:0*
T0*/
_output_shapes
:         `Ъ
2mpn/sequential_1/layer_normalization_5/ones/packedPack0mpn/sequential_1/layer_normalization_5/mul_1:z:0*
N*
T0*
_output_shapes
:v
1mpn/sequential_1/layer_normalization_5/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?┌
+mpn/sequential_1/layer_normalization_5/onesFill;mpn/sequential_1/layer_normalization_5/ones/packed:output:0:mpn/sequential_1/layer_normalization_5/ones/Const:output:0*
T0*#
_output_shapes
:         Ы
3mpn/sequential_1/layer_normalization_5/zeros/packedPack0mpn/sequential_1/layer_normalization_5/mul_1:z:0*
N*
T0*
_output_shapes
:w
2mpn/sequential_1/layer_normalization_5/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ▌
,mpn/sequential_1/layer_normalization_5/zerosFill<mpn/sequential_1/layer_normalization_5/zeros/packed:output:0;mpn/sequential_1/layer_normalization_5/zeros/Const:output:0*
T0*#
_output_shapes
:         o
,mpn/sequential_1/layer_normalization_5/ConstConst*
_output_shapes
: *
dtype0*
valueB q
.mpn/sequential_1/layer_normalization_5/Const_1Const*
_output_shapes
: *
dtype0*
valueB Н
7mpn/sequential_1/layer_normalization_5/FusedBatchNormV3FusedBatchNormV37mpn/sequential_1/layer_normalization_5/Reshape:output:04mpn/sequential_1/layer_normalization_5/ones:output:05mpn/sequential_1/layer_normalization_5/zeros:output:05mpn/sequential_1/layer_normalization_5/Const:output:07mpn/sequential_1/layer_normalization_5/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:         `:         :         :         :         :*
data_formatNCHW*
epsilon%oГ:ю
0mpn/sequential_1/layer_normalization_5/Reshape_1Reshape;mpn/sequential_1/layer_normalization_5/FusedBatchNormV3:y:05mpn/sequential_1/layer_normalization_5/Shape:output:0*
T0*4
_output_shapes"
 :                  `╝
;mpn/sequential_1/layer_normalization_5/mul_3/ReadVariableOpReadVariableOpDmpn_sequential_1_layer_normalization_5_mul_3_readvariableop_resource*
_output_shapes
:`*
dtype0Є
,mpn/sequential_1/layer_normalization_5/mul_3Mul9mpn/sequential_1/layer_normalization_5/Reshape_1:output:0Cmpn/sequential_1/layer_normalization_5/mul_3/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `╕
9mpn/sequential_1/layer_normalization_5/add/ReadVariableOpReadVariableOpBmpn_sequential_1_layer_normalization_5_add_readvariableop_resource*
_output_shapes
:`*
dtype0ч
*mpn/sequential_1/layer_normalization_5/addAddV20mpn/sequential_1/layer_normalization_5/mul_3:z:0Ampn/sequential_1/layer_normalization_5/add/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `Ъ
(node_prediction/Tensordot/ReadVariableOpReadVariableOp1node_prediction_tensordot_readvariableop_resource*
_output_shapes

:`*
dtype0h
node_prediction/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:o
node_prediction/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       }
node_prediction/Tensordot/ShapeShape.mpn/sequential_1/layer_normalization_5/add:z:0*
T0*
_output_shapes
:i
'node_prediction/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : √
"node_prediction/Tensordot/GatherV2GatherV2(node_prediction/Tensordot/Shape:output:0'node_prediction/Tensordot/free:output:00node_prediction/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:k
)node_prediction/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B :  
$node_prediction/Tensordot/GatherV2_1GatherV2(node_prediction/Tensordot/Shape:output:0'node_prediction/Tensordot/axes:output:02node_prediction/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:i
node_prediction/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: Ю
node_prediction/Tensordot/ProdProd+node_prediction/Tensordot/GatherV2:output:0(node_prediction/Tensordot/Const:output:0*
T0*
_output_shapes
: k
!node_prediction/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: д
 node_prediction/Tensordot/Prod_1Prod-node_prediction/Tensordot/GatherV2_1:output:0*node_prediction/Tensordot/Const_1:output:0*
T0*
_output_shapes
: g
%node_prediction/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ▄
 node_prediction/Tensordot/concatConcatV2'node_prediction/Tensordot/free:output:0'node_prediction/Tensordot/axes:output:0.node_prediction/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:й
node_prediction/Tensordot/stackPack'node_prediction/Tensordot/Prod:output:0)node_prediction/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:╩
#node_prediction/Tensordot/transpose	Transpose.mpn/sequential_1/layer_normalization_5/add:z:0)node_prediction/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :                  `║
!node_prediction/Tensordot/ReshapeReshape'node_prediction/Tensordot/transpose:y:0(node_prediction/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  ║
 node_prediction/Tensordot/MatMulMatMul*node_prediction/Tensordot/Reshape:output:00node_prediction/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         k
!node_prediction/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:i
'node_prediction/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ч
"node_prediction/Tensordot/concat_1ConcatV2+node_prediction/Tensordot/GatherV2:output:0*node_prediction/Tensordot/Const_2:output:00node_prediction/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:╝
node_prediction/TensordotReshape*node_prediction/Tensordot/MatMul:product:0+node_prediction/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  Т
&node_prediction/BiasAdd/ReadVariableOpReadVariableOp/node_prediction_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0╡
node_prediction/BiasAddBiasAdd"node_prediction/Tensordot:output:0.node_prediction/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  |
IdentityIdentity node_prediction/BiasAdd:output:0^NoOp*
T0*4
_output_shapes"
 :                  ░	
NoOpNoOp!^edge_ide1/BiasAdd/ReadVariableOp#^edge_ide1/Tensordot/ReadVariableOp!^edge_ide2/BiasAdd/ReadVariableOp#^edge_ide2/Tensordot/ReadVariableOp'^layer_normalization/add/ReadVariableOp)^layer_normalization/mul_3/ReadVariableOp)^layer_normalization_1/add/ReadVariableOp+^layer_normalization_1/mul_3/ReadVariableOp)^layer_normalization_2/add/ReadVariableOp+^layer_normalization_2/mul_3/ReadVariableOp)^layer_normalization_3/add/ReadVariableOp+^layer_normalization_3/mul_3/ReadVariableOp#^mpn/dense_1/BiasAdd/ReadVariableOp%^mpn/dense_1/Tensordot/ReadVariableOp,^mpn/sequential/dense/BiasAdd/ReadVariableOp.^mpn/sequential/dense/Tensordot/ReadVariableOp8^mpn/sequential/layer_normalization_4/add/ReadVariableOp:^mpn/sequential/layer_normalization_4/mul_3/ReadVariableOp:^mpn/sequential_1/layer_normalization_5/add/ReadVariableOp<^mpn/sequential_1/layer_normalization_5/mul_3/ReadVariableOp!^node_ide1/BiasAdd/ReadVariableOp#^node_ide1/Tensordot/ReadVariableOp!^node_ide2/BiasAdd/ReadVariableOp#^node_ide2/Tensordot/ReadVariableOp'^node_prediction/BiasAdd/ReadVariableOp)^node_prediction/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*╔
_input_shapes╖
┤:                  :                  :                  :                  : : : : : : : : : : : : : : : : : : : : : : : : : : 2D
 edge_ide1/BiasAdd/ReadVariableOp edge_ide1/BiasAdd/ReadVariableOp2H
"edge_ide1/Tensordot/ReadVariableOp"edge_ide1/Tensordot/ReadVariableOp2D
 edge_ide2/BiasAdd/ReadVariableOp edge_ide2/BiasAdd/ReadVariableOp2H
"edge_ide2/Tensordot/ReadVariableOp"edge_ide2/Tensordot/ReadVariableOp2P
&layer_normalization/add/ReadVariableOp&layer_normalization/add/ReadVariableOp2T
(layer_normalization/mul_3/ReadVariableOp(layer_normalization/mul_3/ReadVariableOp2T
(layer_normalization_1/add/ReadVariableOp(layer_normalization_1/add/ReadVariableOp2X
*layer_normalization_1/mul_3/ReadVariableOp*layer_normalization_1/mul_3/ReadVariableOp2T
(layer_normalization_2/add/ReadVariableOp(layer_normalization_2/add/ReadVariableOp2X
*layer_normalization_2/mul_3/ReadVariableOp*layer_normalization_2/mul_3/ReadVariableOp2T
(layer_normalization_3/add/ReadVariableOp(layer_normalization_3/add/ReadVariableOp2X
*layer_normalization_3/mul_3/ReadVariableOp*layer_normalization_3/mul_3/ReadVariableOp2H
"mpn/dense_1/BiasAdd/ReadVariableOp"mpn/dense_1/BiasAdd/ReadVariableOp2L
$mpn/dense_1/Tensordot/ReadVariableOp$mpn/dense_1/Tensordot/ReadVariableOp2Z
+mpn/sequential/dense/BiasAdd/ReadVariableOp+mpn/sequential/dense/BiasAdd/ReadVariableOp2^
-mpn/sequential/dense/Tensordot/ReadVariableOp-mpn/sequential/dense/Tensordot/ReadVariableOp2r
7mpn/sequential/layer_normalization_4/add/ReadVariableOp7mpn/sequential/layer_normalization_4/add/ReadVariableOp2v
9mpn/sequential/layer_normalization_4/mul_3/ReadVariableOp9mpn/sequential/layer_normalization_4/mul_3/ReadVariableOp2v
9mpn/sequential_1/layer_normalization_5/add/ReadVariableOp9mpn/sequential_1/layer_normalization_5/add/ReadVariableOp2z
;mpn/sequential_1/layer_normalization_5/mul_3/ReadVariableOp;mpn/sequential_1/layer_normalization_5/mul_3/ReadVariableOp2D
 node_ide1/BiasAdd/ReadVariableOp node_ide1/BiasAdd/ReadVariableOp2H
"node_ide1/Tensordot/ReadVariableOp"node_ide1/Tensordot/ReadVariableOp2D
 node_ide2/BiasAdd/ReadVariableOp node_ide2/BiasAdd/ReadVariableOp2H
"node_ide2/Tensordot/ReadVariableOp"node_ide2/Tensordot/ReadVariableOp2P
&node_prediction/BiasAdd/ReadVariableOp&node_prediction/BiasAdd/ReadVariableOp2T
(node_prediction/Tensordot/ReadVariableOp(node_prediction/Tensordot/ReadVariableOp:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_0:^Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_1:^Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_2:^Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_3
╦
B
&__inference_lambda_layer_call_fn_18826

inputs
identity╣
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_lambda_layer_call_and_return_conditional_losses_16872m
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :                  @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :                  @:\ X
4
_output_shapes"
 :                  @
 
_user_specified_nameinputs
╝	
]
A__inference_lambda_layer_call_and_return_conditional_losses_15364

inputs
identityO

Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?k
Gelu/mulMulGelu/mul/x:output:0inputs*
T0*4
_output_shapes"
 :                  `P
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *є╡?t
Gelu/truedivRealDivinputsGelu/Cast/x:output:0*
T0*4
_output_shapes"
 :                  ``
Gelu/ErfErfGelu/truediv:z:0*
T0*4
_output_shapes"
 :                  `O

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?s
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*4
_output_shapes"
 :                  `l

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*4
_output_shapes"
 :                  `c
IdentityIdentityGelu/mul_1:z:0*
T0*4
_output_shapes"
 :                  `"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :                  `:\ X
4
_output_shapes"
 :                  `
 
_user_specified_nameinputs
МX
ё
E__inference_sequential_layer_call_and_return_conditional_losses_20072

inputs:
'dense_tensordot_readvariableop_resource:	а`3
%dense_biasadd_readvariableop_resource:`A
3layer_normalization_4_mul_3_readvariableop_resource:`?
1layer_normalization_4_add_readvariableop_resource:`
identityИвdense/BiasAdd/ReadVariableOpвdense/Tensordot/ReadVariableOpв(layer_normalization_4/add/ReadVariableOpв*layer_normalization_4/mul_3/ReadVariableOpЗ
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes
:	а`*
dtype0^
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:e
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       K
dense/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:_
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╙
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ╫
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:_
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: А
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: a
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Ж
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ]
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ┤
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Л
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:П
dense/Tensordot/transpose	Transposeinputsdense/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:                  аЬ
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  Ь
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `a
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:`_
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ┐
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Ю
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  `~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0Ч
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `V
lambda/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Й
lambda/Gelu/mulMullambda/Gelu/mul/x:output:0dense/BiasAdd:output:0*
T0*4
_output_shapes"
 :                  `W
lambda/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *є╡?Т
lambda/Gelu/truedivRealDivdense/BiasAdd:output:0lambda/Gelu/Cast/x:output:0*
T0*4
_output_shapes"
 :                  `n
lambda/Gelu/ErfErflambda/Gelu/truediv:z:0*
T0*4
_output_shapes"
 :                  `V
lambda/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?И
lambda/Gelu/addAddV2lambda/Gelu/add/x:output:0lambda/Gelu/Erf:y:0*
T0*4
_output_shapes"
 :                  `Б
lambda/Gelu/mul_1Mullambda/Gelu/mul:z:0lambda/Gelu/add:z:0*
T0*4
_output_shapes"
 :                  ``
layer_normalization_4/ShapeShapelambda/Gelu/mul_1:z:0*
T0*
_output_shapes
:s
)layer_normalization_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+layer_normalization_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┐
#layer_normalization_4/strided_sliceStridedSlice$layer_normalization_4/Shape:output:02layer_normalization_4/strided_slice/stack:output:04layer_normalization_4/strided_slice/stack_1:output:04layer_normalization_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization_4/mul/xConst*
_output_shapes
: *
dtype0*
value	B :Х
layer_normalization_4/mulMul$layer_normalization_4/mul/x:output:0,layer_normalization_4/strided_slice:output:0*
T0*
_output_shapes
: u
+layer_normalization_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╟
%layer_normalization_4/strided_slice_1StridedSlice$layer_normalization_4/Shape:output:04layer_normalization_4/strided_slice_1/stack:output:06layer_normalization_4/strided_slice_1/stack_1:output:06layer_normalization_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskТ
layer_normalization_4/mul_1Mullayer_normalization_4/mul:z:0.layer_normalization_4/strided_slice_1:output:0*
T0*
_output_shapes
: u
+layer_normalization_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╟
%layer_normalization_4/strided_slice_2StridedSlice$layer_normalization_4/Shape:output:04layer_normalization_4/strided_slice_2/stack:output:06layer_normalization_4/strided_slice_2/stack_1:output:06layer_normalization_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
layer_normalization_4/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :Ы
layer_normalization_4/mul_2Mul&layer_normalization_4/mul_2/x:output:0.layer_normalization_4/strided_slice_2:output:0*
T0*
_output_shapes
: g
%layer_normalization_4/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :g
%layer_normalization_4/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :√
#layer_normalization_4/Reshape/shapePack.layer_normalization_4/Reshape/shape/0:output:0layer_normalization_4/mul_1:z:0layer_normalization_4/mul_2:z:0.layer_normalization_4/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:з
layer_normalization_4/ReshapeReshapelambda/Gelu/mul_1:z:0,layer_normalization_4/Reshape/shape:output:0*
T0*/
_output_shapes
:         `x
!layer_normalization_4/ones/packedPacklayer_normalization_4/mul_1:z:0*
N*
T0*
_output_shapes
:e
 layer_normalization_4/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?з
layer_normalization_4/onesFill*layer_normalization_4/ones/packed:output:0)layer_normalization_4/ones/Const:output:0*
T0*#
_output_shapes
:         y
"layer_normalization_4/zeros/packedPacklayer_normalization_4/mul_1:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    к
layer_normalization_4/zerosFill+layer_normalization_4/zeros/packed:output:0*layer_normalization_4/zeros/Const:output:0*
T0*#
_output_shapes
:         ^
layer_normalization_4/ConstConst*
_output_shapes
: *
dtype0*
valueB `
layer_normalization_4/Const_1Const*
_output_shapes
: *
dtype0*
valueB з
&layer_normalization_4/FusedBatchNormV3FusedBatchNormV3&layer_normalization_4/Reshape:output:0#layer_normalization_4/ones:output:0$layer_normalization_4/zeros:output:0$layer_normalization_4/Const:output:0&layer_normalization_4/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:         `:         :         :         :         :*
data_formatNCHW*
epsilon%oГ:╗
layer_normalization_4/Reshape_1Reshape*layer_normalization_4/FusedBatchNormV3:y:0$layer_normalization_4/Shape:output:0*
T0*4
_output_shapes"
 :                  `Ъ
*layer_normalization_4/mul_3/ReadVariableOpReadVariableOp3layer_normalization_4_mul_3_readvariableop_resource*
_output_shapes
:`*
dtype0┐
layer_normalization_4/mul_3Mul(layer_normalization_4/Reshape_1:output:02layer_normalization_4/mul_3/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `Ц
(layer_normalization_4/add/ReadVariableOpReadVariableOp1layer_normalization_4_add_readvariableop_resource*
_output_shapes
:`*
dtype0┤
layer_normalization_4/addAddV2layer_normalization_4/mul_3:z:00layer_normalization_4/add/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `y
IdentityIdentitylayer_normalization_4/add:z:0^NoOp*
T0*4
_output_shapes"
 :                  `▐
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp)^layer_normalization_4/add/ReadVariableOp+^layer_normalization_4/mul_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):                  а: : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2T
(layer_normalization_4/add/ReadVariableOp(layer_normalization_4/add/ReadVariableOp2X
*layer_normalization_4/mul_3/ReadVariableOp*layer_normalization_4/mul_3/ReadVariableOp:] Y
5
_output_shapes#
!:                  а
 
_user_specified_nameinputs
К
Ю
5__inference_layer_normalization_4_layer_call_fn_20248

inputs
unknown:`
	unknown_0:`
identityИвStatefulPartitionedCallЄ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  `*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_layer_normalization_4_layer_call_and_return_conditional_losses_15413|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  ``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:                  `: : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :                  `
 
_user_specified_nameinputs
Г
√
D__inference_edge_ide2_layer_call_and_return_conditional_losses_15966

inputs3
!tensordot_readvariableop_resource:@`-
biasadd_readvariableop_resource:`
identityИвBiasAdd/ReadVariableOpвTensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@`*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ┐
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ь
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:В
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*4
_output_shapes"
 :                  @К
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:                  К
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:`Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : з
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:М
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  `r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:`*
dtype0Е
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `l
IdentityIdentityBiasAdd:output:0^NoOp*
T0*4
_output_shapes"
 :                  `z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:                  @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:\ X
4
_output_shapes"
 :                  @
 
_user_specified_nameinputs
╝	
]
A__inference_lambda_layer_call_and_return_conditional_losses_15754

inputs
identityO

Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?k
Gelu/mulMulGelu/mul/x:output:0inputs*
T0*4
_output_shapes"
 :                  @P
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *є╡?t
Gelu/truedivRealDivinputsGelu/Cast/x:output:0*
T0*4
_output_shapes"
 :                  @`
Gelu/ErfErfGelu/truediv:z:0*
T0*4
_output_shapes"
 :                  @O

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?s
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*4
_output_shapes"
 :                  @l

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*4
_output_shapes"
 :                  @c
IdentityIdentityGelu/mul_1:z:0*
T0*4
_output_shapes"
 :                  @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :                  @:\ X
4
_output_shapes"
 :                  @
 
_user_specified_nameinputs
й
╩
E__inference_sequential_layer_call_and_return_conditional_losses_15543
dense_input
dense_15531:	а`
dense_15533:`)
layer_normalization_4_15537:`)
layer_normalization_4_15539:`
identityИвdense/StatefulPartitionedCallв-layer_normalization_4/StatefulPartitionedCallє
dense/StatefulPartitionedCallStatefulPartitionedCalldense_inputdense_15531dense_15533*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  `*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_15346р
lambda/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  `* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_lambda_layer_call_and_return_conditional_losses_15364╟
-layer_normalization_4/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0layer_normalization_4_15537layer_normalization_4_15539*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  `*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_layer_normalization_4_layer_call_and_return_conditional_losses_15413Т
IdentityIdentity6layer_normalization_4/StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  `Ц
NoOpNoOp^dense/StatefulPartitionedCall.^layer_normalization_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):                  а: : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2^
-layer_normalization_4/StatefulPartitionedCall-layer_normalization_4/StatefulPartitionedCall:b ^
5
_output_shapes#
!:                  а
%
_user_specified_namedense_input
Г
√
D__inference_edge_ide2_layer_call_and_return_conditional_losses_19103

inputs3
!tensordot_readvariableop_resource:@`-
biasadd_readvariableop_resource:`
identityИвBiasAdd/ReadVariableOpвTensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@`*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ┐
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ь
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:В
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*4
_output_shapes"
 :                  @К
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:                  К
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:`Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : з
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:М
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  `r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:`*
dtype0Е
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `l
IdentityIdentityBiasAdd:output:0^NoOp*
T0*4
_output_shapes"
 :                  `z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:                  @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:\ X
4
_output_shapes"
 :                  @
 
_user_specified_nameinputs
Д
°
@__inference_dense_layer_call_and_return_conditional_losses_15346

inputs4
!tensordot_readvariableop_resource:	а`-
biasadd_readvariableop_resource:`
identityИвBiasAdd/ReadVariableOpвTensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	а`*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ┐
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ь
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Г
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*5
_output_shapes#
!:                  аК
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:                  К
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:`Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : з
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:М
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  `r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:`*
dtype0Е
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `l
IdentityIdentityBiasAdd:output:0^NoOp*
T0*4
_output_shapes"
 :                  `z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:                  а: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:] Y
5
_output_shapes#
!:                  а
 
_user_specified_nameinputs
а
ж
#__inference_mpn_layer_call_fn_19281
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
unknown:	а`
	unknown_0:`
	unknown_1:`
	unknown_2:`
	unknown_3:	└`
	unknown_4:`
	unknown_5:`
	unknown_6:`
identity

identity_1

identity_2

identity_3

identity_4ИвStatefulPartitionedCallу
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2*
Tout	
2*
_collective_manager_ids
 *╢
_output_shapesг
а:                  `:                  `:                  :                  :                  **
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8В *G
fBR@
>__inference_mpn_layer_call_and_return_conditional_losses_16786|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  `~

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*4
_output_shapes"
 :                  `~

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*4
_output_shapes"
 :                  ~

Identity_3Identity StatefulPartitionedCall:output:3^NoOp*
T0*4
_output_shapes"
 :                  ~

Identity_4Identity StatefulPartitionedCall:output:4^NoOp*
T0*4
_output_shapes"
 :                  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*┼
_input_shapes│
░:                  `:                  `:                  :                  :                  : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :                  `
"
_user_specified_name
inputs_0:^Z
4
_output_shapes"
 :                  `
"
_user_specified_name
inputs_1:^Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_2:^Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_3:^Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_4
вМ
Е	
>__inference_mpn_layer_call_and_return_conditional_losses_19563
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4E
2sequential_dense_tensordot_readvariableop_resource:	а`>
0sequential_dense_biasadd_readvariableop_resource:`L
>sequential_layer_normalization_4_mul_3_readvariableop_resource:`J
<sequential_layer_normalization_4_add_readvariableop_resource:`<
)dense_1_tensordot_readvariableop_resource:	└`5
'dense_1_biasadd_readvariableop_resource:`N
@sequential_1_layer_normalization_5_mul_3_readvariableop_resource:`L
>sequential_1_layer_normalization_5_add_readvariableop_resource:`
identity

identity_1

identity_2

identity_3

identity_4Ивdense_1/BiasAdd/ReadVariableOpв dense_1/Tensordot/ReadVariableOpв'sequential/dense/BiasAdd/ReadVariableOpв)sequential/dense/Tensordot/ReadVariableOpв3sequential/layer_normalization_4/add/ReadVariableOpв5sequential/layer_normalization_4/mul_3/ReadVariableOpв5sequential_1/layer_normalization_5/add/ReadVariableOpв7sequential_1/layer_normalization_5/mul_3/ReadVariableOp=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Shape_1Shapeinputs_2*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Shape_2Shapeinputs_0*
T0*
_output_shapes
:_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_2StridedSliceShape_2:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskO
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :╕
GatherV2GatherV2inputs_0inputs_2GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*8
_output_shapes&
$:"                  `*

batch_dimsR
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :└С
Reshape/shapePackstrided_slice_2:output:0strided_slice_1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:}
ReshapeReshapeGatherV2:output:0Reshape/shape:output:0*
T0*5
_output_shapes#
!:                  └V
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
         Н
concatConcatV2Reshape:output:0inputs_1concat/axis:output:0*
N*
T0*5
_output_shapes#
!:                  аЭ
)sequential/dense/Tensordot/ReadVariableOpReadVariableOp2sequential_dense_tensordot_readvariableop_resource*
_output_shapes
:	а`*
dtype0i
sequential/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:p
sequential/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       _
 sequential/dense/Tensordot/ShapeShapeconcat:output:0*
T0*
_output_shapes
:j
(sequential/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :  
#sequential/dense/Tensordot/GatherV2GatherV2)sequential/dense/Tensordot/Shape:output:0(sequential/dense/Tensordot/free:output:01sequential/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
*sequential/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Г
%sequential/dense/Tensordot/GatherV2_1GatherV2)sequential/dense/Tensordot/Shape:output:0(sequential/dense/Tensordot/axes:output:03sequential/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:j
 sequential/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: б
sequential/dense/Tensordot/ProdProd,sequential/dense/Tensordot/GatherV2:output:0)sequential/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: l
"sequential/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: з
!sequential/dense/Tensordot/Prod_1Prod.sequential/dense/Tensordot/GatherV2_1:output:0+sequential/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: h
&sequential/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : р
!sequential/dense/Tensordot/concatConcatV2(sequential/dense/Tensordot/free:output:0(sequential/dense/Tensordot/axes:output:0/sequential/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:м
 sequential/dense/Tensordot/stackPack(sequential/dense/Tensordot/Prod:output:0*sequential/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:о
$sequential/dense/Tensordot/transpose	Transposeconcat:output:0*sequential/dense/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:                  а╜
"sequential/dense/Tensordot/ReshapeReshape(sequential/dense/Tensordot/transpose:y:0)sequential/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  ╜
!sequential/dense/Tensordot/MatMulMatMul+sequential/dense/Tensordot/Reshape:output:01sequential/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `l
"sequential/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:`j
(sequential/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ы
#sequential/dense/Tensordot/concat_1ConcatV2,sequential/dense/Tensordot/GatherV2:output:0+sequential/dense/Tensordot/Const_2:output:01sequential/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:┐
sequential/dense/TensordotReshape+sequential/dense/Tensordot/MatMul:product:0,sequential/dense/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  `Ф
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0╕
sequential/dense/BiasAddBiasAdd#sequential/dense/Tensordot:output:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `a
sequential/lambda/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?к
sequential/lambda/Gelu/mulMul%sequential/lambda/Gelu/mul/x:output:0!sequential/dense/BiasAdd:output:0*
T0*4
_output_shapes"
 :                  `b
sequential/lambda/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *є╡?│
sequential/lambda/Gelu/truedivRealDiv!sequential/dense/BiasAdd:output:0&sequential/lambda/Gelu/Cast/x:output:0*
T0*4
_output_shapes"
 :                  `Д
sequential/lambda/Gelu/ErfErf"sequential/lambda/Gelu/truediv:z:0*
T0*4
_output_shapes"
 :                  `a
sequential/lambda/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?й
sequential/lambda/Gelu/addAddV2%sequential/lambda/Gelu/add/x:output:0sequential/lambda/Gelu/Erf:y:0*
T0*4
_output_shapes"
 :                  `в
sequential/lambda/Gelu/mul_1Mulsequential/lambda/Gelu/mul:z:0sequential/lambda/Gelu/add:z:0*
T0*4
_output_shapes"
 :                  `v
&sequential/layer_normalization_4/ShapeShape sequential/lambda/Gelu/mul_1:z:0*
T0*
_output_shapes
:~
4sequential/layer_normalization_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: А
6sequential/layer_normalization_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:А
6sequential/layer_normalization_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ў
.sequential/layer_normalization_4/strided_sliceStridedSlice/sequential/layer_normalization_4/Shape:output:0=sequential/layer_normalization_4/strided_slice/stack:output:0?sequential/layer_normalization_4/strided_slice/stack_1:output:0?sequential/layer_normalization_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
&sequential/layer_normalization_4/mul/xConst*
_output_shapes
: *
dtype0*
value	B :╢
$sequential/layer_normalization_4/mulMul/sequential/layer_normalization_4/mul/x:output:07sequential/layer_normalization_4/strided_slice:output:0*
T0*
_output_shapes
: А
6sequential/layer_normalization_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:В
8sequential/layer_normalization_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:В
8sequential/layer_normalization_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:■
0sequential/layer_normalization_4/strided_slice_1StridedSlice/sequential/layer_normalization_4/Shape:output:0?sequential/layer_normalization_4/strided_slice_1/stack:output:0Asequential/layer_normalization_4/strided_slice_1/stack_1:output:0Asequential/layer_normalization_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask│
&sequential/layer_normalization_4/mul_1Mul(sequential/layer_normalization_4/mul:z:09sequential/layer_normalization_4/strided_slice_1:output:0*
T0*
_output_shapes
: А
6sequential/layer_normalization_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:В
8sequential/layer_normalization_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:В
8sequential/layer_normalization_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:■
0sequential/layer_normalization_4/strided_slice_2StridedSlice/sequential/layer_normalization_4/Shape:output:0?sequential/layer_normalization_4/strided_slice_2/stack:output:0Asequential/layer_normalization_4/strided_slice_2/stack_1:output:0Asequential/layer_normalization_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(sequential/layer_normalization_4/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :╝
&sequential/layer_normalization_4/mul_2Mul1sequential/layer_normalization_4/mul_2/x:output:09sequential/layer_normalization_4/strided_slice_2:output:0*
T0*
_output_shapes
: r
0sequential/layer_normalization_4/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :r
0sequential/layer_normalization_4/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :▓
.sequential/layer_normalization_4/Reshape/shapePack9sequential/layer_normalization_4/Reshape/shape/0:output:0*sequential/layer_normalization_4/mul_1:z:0*sequential/layer_normalization_4/mul_2:z:09sequential/layer_normalization_4/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:╚
(sequential/layer_normalization_4/ReshapeReshape sequential/lambda/Gelu/mul_1:z:07sequential/layer_normalization_4/Reshape/shape:output:0*
T0*/
_output_shapes
:         `О
,sequential/layer_normalization_4/ones/packedPack*sequential/layer_normalization_4/mul_1:z:0*
N*
T0*
_output_shapes
:p
+sequential/layer_normalization_4/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╚
%sequential/layer_normalization_4/onesFill5sequential/layer_normalization_4/ones/packed:output:04sequential/layer_normalization_4/ones/Const:output:0*
T0*#
_output_shapes
:         П
-sequential/layer_normalization_4/zeros/packedPack*sequential/layer_normalization_4/mul_1:z:0*
N*
T0*
_output_shapes
:q
,sequential/layer_normalization_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ╦
&sequential/layer_normalization_4/zerosFill6sequential/layer_normalization_4/zeros/packed:output:05sequential/layer_normalization_4/zeros/Const:output:0*
T0*#
_output_shapes
:         i
&sequential/layer_normalization_4/ConstConst*
_output_shapes
: *
dtype0*
valueB k
(sequential/layer_normalization_4/Const_1Const*
_output_shapes
: *
dtype0*
valueB щ
1sequential/layer_normalization_4/FusedBatchNormV3FusedBatchNormV31sequential/layer_normalization_4/Reshape:output:0.sequential/layer_normalization_4/ones:output:0/sequential/layer_normalization_4/zeros:output:0/sequential/layer_normalization_4/Const:output:01sequential/layer_normalization_4/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:         `:         :         :         :         :*
data_formatNCHW*
epsilon%oГ:▄
*sequential/layer_normalization_4/Reshape_1Reshape5sequential/layer_normalization_4/FusedBatchNormV3:y:0/sequential/layer_normalization_4/Shape:output:0*
T0*4
_output_shapes"
 :                  `░
5sequential/layer_normalization_4/mul_3/ReadVariableOpReadVariableOp>sequential_layer_normalization_4_mul_3_readvariableop_resource*
_output_shapes
:`*
dtype0р
&sequential/layer_normalization_4/mul_3Mul3sequential/layer_normalization_4/Reshape_1:output:0=sequential/layer_normalization_4/mul_3/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `м
3sequential/layer_normalization_4/add/ReadVariableOpReadVariableOp<sequential_layer_normalization_4_add_readvariableop_resource*
_output_shapes
:`*
dtype0╒
$sequential/layer_normalization_4/addAddV2*sequential/layer_normalization_4/mul_3:z:0;sequential/layer_normalization_4/add/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `}
mulMul(sequential/layer_normalization_4/add:z:0inputs_3*
T0*4
_output_shapes"
 :                  `P
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :`s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         `A

scan/ShapeShapemul:z:0*
T0*
_output_shapes
:b
scan/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: d
scan/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:d
scan/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ъ
scan/strided_sliceStridedSlicescan/Shape:output:0!scan/strided_slice/stack:output:0#scan/strided_slice/stack_1:output:0#scan/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskq
 scan/TensorArrayV2/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    `   ┴
scan/TensorArrayV2TensorListReserve)scan/TensorArrayV2/element_shape:output:0scan/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥s
"scan/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ┼
scan/TensorArrayV2_1TensorListReserve+scan/TensorArrayV2_1/element_shape:output:0scan/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╧s
"scan/TensorArrayV2_2/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ┼
scan/TensorArrayV2_2TensorListReserve+scan/TensorArrayV2_2/element_shape:output:0scan/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Л
:scan/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    `   ф
,scan/TensorArrayUnstack/TensorListFromTensorTensorListFromTensormul:z:0Cscan/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Н
<scan/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       щ
.scan/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensorinputs_2Escan/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╧Н
<scan/TensorArrayUnstack_2/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       щ
.scan/TensorArrayUnstack_2/TensorListFromTensorTensorListFromTensorinputs_4Escan/TensorArrayUnstack_2/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥s
"scan/TensorArrayV2_3/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    `   ┼
scan/TensorArrayV2_3TensorListReserve+scan/TensorArrayV2_3/element_shape:output:0scan/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥L

scan/ConstConst*
_output_shapes
: *
dtype0*
value	B : Y
scan/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ▀

scan/whileStatelessWhile scan/while/loop_counter:output:0scan/strided_slice:output:0scan/Const:output:0zeros:output:0scan/TensorArrayV2_3:handle:0scan/strided_slice:output:0<scan/TensorArrayUnstack/TensorListFromTensor:output_handle:0>scan/TensorArrayUnstack_1/TensorListFromTensor:output_handle:0>scan/TensorArrayUnstack_2/TensorListFromTensor:output_handle:0strided_slice:output:0*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : :         `: : : : : : * 
_read_only_resource_inputs
 *
_stateful_parallelism( *!
bodyR
scan_while_body_19413*!
condR
scan_while_cond_19412*8
output_shapes'
%: : : :         `: : : : : : Ж
5scan/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    `   ┌
'scan/TensorArrayV2Stack/TensorListStackTensorListStackscan/while:output:4>scan/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  `*
element_dtype0_
lambda_1/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
         ┐
lambda_1/concatConcatV2inputs_00scan/TensorArrayV2Stack/TensorListStack:tensor:0lambda_1/concat/axis:output:0*
N*
T0*5
_output_shapes#
!:                  └Л
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource*
_output_shapes
:	└`*
dtype0`
dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       _
dense_1/Tensordot/ShapeShapelambda_1/concat:output:0*
T0*
_output_shapes
:a
dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : █
dense_1/Tensordot/GatherV2GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/free:output:0(dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ▀
dense_1/Tensordot/GatherV2_1GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/axes:output:0*dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: Ж
dense_1/Tensordot/ProdProd#dense_1/Tensordot/GatherV2:output:0 dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: М
dense_1/Tensordot/Prod_1Prod%dense_1/Tensordot/GatherV2_1:output:0"dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ╝
dense_1/Tensordot/concatConcatV2dense_1/Tensordot/free:output:0dense_1/Tensordot/axes:output:0&dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:С
dense_1/Tensordot/stackPackdense_1/Tensordot/Prod:output:0!dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:е
dense_1/Tensordot/transpose	Transposelambda_1/concat:output:0!dense_1/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:                  └в
dense_1/Tensordot/ReshapeReshapedense_1/Tensordot/transpose:y:0 dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  в
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0(dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `c
dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:`a
dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ╟
dense_1/Tensordot/concat_1ConcatV2#dense_1/Tensordot/GatherV2:output:0"dense_1/Tensordot/Const_2:output:0(dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:д
dense_1/TensordotReshape"dense_1/Tensordot/MatMul:product:0#dense_1/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  `В
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0Э
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `c
sequential_1/lambda/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?е
sequential_1/lambda/Gelu/mulMul'sequential_1/lambda/Gelu/mul/x:output:0dense_1/BiasAdd:output:0*
T0*4
_output_shapes"
 :                  `d
sequential_1/lambda/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *є╡?о
 sequential_1/lambda/Gelu/truedivRealDivdense_1/BiasAdd:output:0(sequential_1/lambda/Gelu/Cast/x:output:0*
T0*4
_output_shapes"
 :                  `И
sequential_1/lambda/Gelu/ErfErf$sequential_1/lambda/Gelu/truediv:z:0*
T0*4
_output_shapes"
 :                  `c
sequential_1/lambda/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?п
sequential_1/lambda/Gelu/addAddV2'sequential_1/lambda/Gelu/add/x:output:0 sequential_1/lambda/Gelu/Erf:y:0*
T0*4
_output_shapes"
 :                  `и
sequential_1/lambda/Gelu/mul_1Mul sequential_1/lambda/Gelu/mul:z:0 sequential_1/lambda/Gelu/add:z:0*
T0*4
_output_shapes"
 :                  `z
(sequential_1/layer_normalization_5/ShapeShape"sequential_1/lambda/Gelu/mul_1:z:0*
T0*
_output_shapes
:А
6sequential_1/layer_normalization_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: В
8sequential_1/layer_normalization_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:В
8sequential_1/layer_normalization_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:А
0sequential_1/layer_normalization_5/strided_sliceStridedSlice1sequential_1/layer_normalization_5/Shape:output:0?sequential_1/layer_normalization_5/strided_slice/stack:output:0Asequential_1/layer_normalization_5/strided_slice/stack_1:output:0Asequential_1/layer_normalization_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(sequential_1/layer_normalization_5/mul/xConst*
_output_shapes
: *
dtype0*
value	B :╝
&sequential_1/layer_normalization_5/mulMul1sequential_1/layer_normalization_5/mul/x:output:09sequential_1/layer_normalization_5/strided_slice:output:0*
T0*
_output_shapes
: В
8sequential_1/layer_normalization_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:Д
:sequential_1/layer_normalization_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Д
:sequential_1/layer_normalization_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
2sequential_1/layer_normalization_5/strided_slice_1StridedSlice1sequential_1/layer_normalization_5/Shape:output:0Asequential_1/layer_normalization_5/strided_slice_1/stack:output:0Csequential_1/layer_normalization_5/strided_slice_1/stack_1:output:0Csequential_1/layer_normalization_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask╣
(sequential_1/layer_normalization_5/mul_1Mul*sequential_1/layer_normalization_5/mul:z:0;sequential_1/layer_normalization_5/strided_slice_1:output:0*
T0*
_output_shapes
: В
8sequential_1/layer_normalization_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:Д
:sequential_1/layer_normalization_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Д
:sequential_1/layer_normalization_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
2sequential_1/layer_normalization_5/strided_slice_2StridedSlice1sequential_1/layer_normalization_5/Shape:output:0Asequential_1/layer_normalization_5/strided_slice_2/stack:output:0Csequential_1/layer_normalization_5/strided_slice_2/stack_1:output:0Csequential_1/layer_normalization_5/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
*sequential_1/layer_normalization_5/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :┬
(sequential_1/layer_normalization_5/mul_2Mul3sequential_1/layer_normalization_5/mul_2/x:output:0;sequential_1/layer_normalization_5/strided_slice_2:output:0*
T0*
_output_shapes
: t
2sequential_1/layer_normalization_5/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :t
2sequential_1/layer_normalization_5/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :╝
0sequential_1/layer_normalization_5/Reshape/shapePack;sequential_1/layer_normalization_5/Reshape/shape/0:output:0,sequential_1/layer_normalization_5/mul_1:z:0,sequential_1/layer_normalization_5/mul_2:z:0;sequential_1/layer_normalization_5/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:╬
*sequential_1/layer_normalization_5/ReshapeReshape"sequential_1/lambda/Gelu/mul_1:z:09sequential_1/layer_normalization_5/Reshape/shape:output:0*
T0*/
_output_shapes
:         `Т
.sequential_1/layer_normalization_5/ones/packedPack,sequential_1/layer_normalization_5/mul_1:z:0*
N*
T0*
_output_shapes
:r
-sequential_1/layer_normalization_5/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╬
'sequential_1/layer_normalization_5/onesFill7sequential_1/layer_normalization_5/ones/packed:output:06sequential_1/layer_normalization_5/ones/Const:output:0*
T0*#
_output_shapes
:         У
/sequential_1/layer_normalization_5/zeros/packedPack,sequential_1/layer_normalization_5/mul_1:z:0*
N*
T0*
_output_shapes
:s
.sequential_1/layer_normalization_5/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ╤
(sequential_1/layer_normalization_5/zerosFill8sequential_1/layer_normalization_5/zeros/packed:output:07sequential_1/layer_normalization_5/zeros/Const:output:0*
T0*#
_output_shapes
:         k
(sequential_1/layer_normalization_5/ConstConst*
_output_shapes
: *
dtype0*
valueB m
*sequential_1/layer_normalization_5/Const_1Const*
_output_shapes
: *
dtype0*
valueB ї
3sequential_1/layer_normalization_5/FusedBatchNormV3FusedBatchNormV33sequential_1/layer_normalization_5/Reshape:output:00sequential_1/layer_normalization_5/ones:output:01sequential_1/layer_normalization_5/zeros:output:01sequential_1/layer_normalization_5/Const:output:03sequential_1/layer_normalization_5/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:         `:         :         :         :         :*
data_formatNCHW*
epsilon%oГ:т
,sequential_1/layer_normalization_5/Reshape_1Reshape7sequential_1/layer_normalization_5/FusedBatchNormV3:y:01sequential_1/layer_normalization_5/Shape:output:0*
T0*4
_output_shapes"
 :                  `┤
7sequential_1/layer_normalization_5/mul_3/ReadVariableOpReadVariableOp@sequential_1_layer_normalization_5_mul_3_readvariableop_resource*
_output_shapes
:`*
dtype0ц
(sequential_1/layer_normalization_5/mul_3Mul5sequential_1/layer_normalization_5/Reshape_1:output:0?sequential_1/layer_normalization_5/mul_3/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `░
5sequential_1/layer_normalization_5/add/ReadVariableOpReadVariableOp>sequential_1_layer_normalization_5_add_readvariableop_resource*
_output_shapes
:`*
dtype0█
&sequential_1/layer_normalization_5/addAddV2,sequential_1/layer_normalization_5/mul_3:z:0=sequential_1/layer_normalization_5/add/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  `Ж
IdentityIdentity*sequential_1/layer_normalization_5/add:z:0^NoOp*
T0*4
_output_shapes"
 :                  `e

Identity_1Identitymul:z:0^NoOp*
T0*4
_output_shapes"
 :                  `f

Identity_2Identityinputs_2^NoOp*
T0*4
_output_shapes"
 :                  f

Identity_3Identityinputs_3^NoOp*
T0*4
_output_shapes"
 :                  f

Identity_4Identityinputs_4^NoOp*
T0*4
_output_shapes"
 :                  └
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp*^sequential/dense/Tensordot/ReadVariableOp4^sequential/layer_normalization_4/add/ReadVariableOp6^sequential/layer_normalization_4/mul_3/ReadVariableOp6^sequential_1/layer_normalization_5/add/ReadVariableOp8^sequential_1/layer_normalization_5/mul_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*┼
_input_shapes│
░:                  `:                  `:                  :                  :                  : : : : : : : : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_1/Tensordot/ReadVariableOp dense_1/Tensordot/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2V
)sequential/dense/Tensordot/ReadVariableOp)sequential/dense/Tensordot/ReadVariableOp2j
3sequential/layer_normalization_4/add/ReadVariableOp3sequential/layer_normalization_4/add/ReadVariableOp2n
5sequential/layer_normalization_4/mul_3/ReadVariableOp5sequential/layer_normalization_4/mul_3/ReadVariableOp2n
5sequential_1/layer_normalization_5/add/ReadVariableOp5sequential_1/layer_normalization_5/add/ReadVariableOp2r
7sequential_1/layer_normalization_5/mul_3/ReadVariableOp7sequential_1/layer_normalization_5/mul_3/ReadVariableOp:^ Z
4
_output_shapes"
 :                  `
"
_user_specified_name
inputs_0:^Z
4
_output_shapes"
 :                  `
"
_user_specified_name
inputs_1:^Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_2:^Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_3:^Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_4
╦
╩
*__inference_sequential_layer_call_fn_19897

inputs
unknown:	а`
	unknown_0:`
	unknown_1:`
	unknown_2:`
identityИвStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  `*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_15420|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  ``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):                  а: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:                  а
 
_user_specified_nameinputs
Г
√
D__inference_node_ide1_layer_call_and_return_conditional_losses_18806

inputs3
!tensordot_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвTensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ┐
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ь
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:В
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*4
_output_shapes"
 :                  К
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:                  К
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : з
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:М
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Е
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  @l
IdentityIdentityBiasAdd:output:0^NoOp*
T0*4
_output_shapes"
 :                  @z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:                  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
К
Ю
5__inference_layer_normalization_3_layer_call_fn_19168

inputs
unknown:`
	unknown_0:`
identityИвStatefulPartitionedCallЄ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  `*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_layer_normalization_3_layer_call_and_return_conditional_losses_16077|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  ``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:                  `: : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :                  `
 
_user_specified_nameinputs"Ж
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*к
serving_defaultЦ
H
input_1=
serving_default_input_1:0                  
H
input_2=
serving_default_input_2:0                  
H
input_3=
serving_default_input_3:0                  
H
input_4=
serving_default_input_4:0                  P
node_prediction=
StatefulPartitionedCall:0                  tensorflow/serving/predict:ЕГ
╦
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer-9
layer_with_weights-6
layer-10
layer_with_weights-7
layer-11
layer-12
layer-13
layer-14
layer_with_weights-8
layer-15
layer_with_weights-9
layer-16
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_network
"
_tf_keras_input_layer
╗
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses

!kernel
"bias"
_tf_keras_layer
е
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses"
_tf_keras_layer
─
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses
/axis
	0gamma
1beta"
_tf_keras_layer
"
_tf_keras_input_layer
╗
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses

8kernel
9bias"
_tf_keras_layer
╗
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses

@kernel
Abias"
_tf_keras_layer
─
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses
Haxis
	Igamma
Jbeta"
_tf_keras_layer
╗
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses

Qkernel
Rbias"
_tf_keras_layer
(
S	keras_api"
_tf_keras_layer
─
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses
Zaxis
	[gamma
\beta"
_tf_keras_layer
─
]	variables
^trainable_variables
_regularization_losses
`	keras_api
a__call__
*b&call_and_return_all_conditional_losses
caxis
	dgamma
ebeta"
_tf_keras_layer
"
_tf_keras_input_layer
(
f	keras_api"
_tf_keras_layer
"
_tf_keras_input_layer
ю
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses
mcombine_layer
nmessage_layer
oupdate_layer
pupdate_norm"
_tf_keras_layer
╗
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
u__call__
*v&call_and_return_all_conditional_losses

wkernel
xbias"
_tf_keras_layer
ч
!0
"1
02
13
84
95
@6
A7
I8
J9
Q10
R11
[12
\13
d14
e15
y16
z17
{18
|19
}20
~21
22
А23
w24
x25"
trackable_list_wrapper
ч
!0
"1
02
13
84
95
@6
A7
I8
J9
Q10
R11
[12
\13
d14
e15
y16
z17
{18
|19
}20
~21
22
А23
w24
x25"
trackable_list_wrapper
 "
trackable_list_wrapper
╧
Бnon_trainable_variables
Вlayers
Гmetrics
 Дlayer_regularization_losses
Еlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
╤
Жtrace_0
Зtrace_1
Иtrace_2
Йtrace_32▐
%__inference_model_layer_call_fn_16468
%__inference_model_layer_call_fn_17471
%__inference_model_layer_call_fn_17531
%__inference_model_layer_call_fn_17179┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЖtrace_0zЗtrace_1zИtrace_2zЙtrace_3
╜
Кtrace_0
Лtrace_1
Мtrace_2
Нtrace_32╩
@__inference_model_layer_call_and_return_conditional_losses_18149
@__inference_model_layer_call_and_return_conditional_losses_18767
@__inference_model_layer_call_and_return_conditional_losses_17263
@__inference_model_layer_call_and_return_conditional_losses_17347┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zКtrace_0zЛtrace_1zМtrace_2zНtrace_3
цBу
 __inference__wrapped_model_15309input_1input_2input_3input_4"Ш
С▓Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
г
О
_variables
П_iterations
Р_learning_rate
С_index_dict
Т
_momentums
У_velocities
Ф_update_step_xla"
experimentalOptimizer
-
Хserving_default"
signature_map
.
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Цnon_trainable_variables
Чlayers
Шmetrics
 Щlayer_regularization_losses
Ъlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses"
_generic_user_object
я
Ыtrace_02╨
)__inference_node_ide1_layer_call_fn_18776в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЫtrace_0
К
Ьtrace_02ы
D__inference_node_ide1_layer_call_and_return_conditional_losses_18806в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЬtrace_0
": @2node_ide1/kernel
:@2node_ide1/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Эnon_trainable_variables
Юlayers
Яmetrics
 аlayer_regularization_losses
бlayer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses"
_generic_user_object
╒
вtrace_0
гtrace_1
дtrace_2
еtrace_32т
&__inference_lambda_layer_call_fn_18811
&__inference_lambda_layer_call_fn_18816
&__inference_lambda_layer_call_fn_18821
&__inference_lambda_layer_call_fn_18826┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsЪ

 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zвtrace_0zгtrace_1zдtrace_2zеtrace_3
┴
жtrace_0
зtrace_1
иtrace_2
йtrace_32╬
A__inference_lambda_layer_call_and_return_conditional_losses_18838
A__inference_lambda_layer_call_and_return_conditional_losses_18850
A__inference_lambda_layer_call_and_return_conditional_losses_18862
A__inference_lambda_layer_call_and_return_conditional_losses_18874┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsЪ

 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zжtrace_0zзtrace_1zиtrace_2zйtrace_3
.
00
11"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
кnon_trainable_variables
лlayers
мmetrics
 нlayer_regularization_losses
оlayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses"
_generic_user_object
∙
пtrace_02┌
3__inference_layer_normalization_layer_call_fn_18883в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zпtrace_0
Ф
░trace_02ї
N__inference_layer_normalization_layer_call_and_return_conditional_losses_18930в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z░trace_0
 "
trackable_list_wrapper
':%@2layer_normalization/gamma
&:$@2layer_normalization/beta
.
80
91"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
▒non_trainable_variables
▓layers
│metrics
 ┤layer_regularization_losses
╡layer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses"
_generic_user_object
я
╢trace_02╨
)__inference_node_ide2_layer_call_fn_18939в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╢trace_0
К
╖trace_02ы
D__inference_node_ide2_layer_call_and_return_conditional_losses_18969в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╖trace_0
": @`2node_ide2/kernel
:`2node_ide2/bias
.
@0
A1"
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
╕non_trainable_variables
╣layers
║metrics
 ╗layer_regularization_losses
╝layer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
я
╜trace_02╨
)__inference_edge_ide1_layer_call_fn_18978в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╜trace_0
К
╛trace_02ы
D__inference_edge_ide1_layer_call_and_return_conditional_losses_19008в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╛trace_0
": @2edge_ide1/kernel
:@2edge_ide1/bias
.
I0
J1"
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
┐non_trainable_variables
└layers
┴metrics
 ┬layer_regularization_losses
├layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
√
─trace_02▄
5__inference_layer_normalization_1_layer_call_fn_19017в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z─trace_0
Ц
┼trace_02ў
P__inference_layer_normalization_1_layer_call_and_return_conditional_losses_19064в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z┼trace_0
 "
trackable_list_wrapper
):'@2layer_normalization_1/gamma
(:&@2layer_normalization_1/beta
.
Q0
R1"
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
╞non_trainable_variables
╟layers
╚metrics
 ╔layer_regularization_losses
╩layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
я
╦trace_02╨
)__inference_edge_ide2_layer_call_fn_19073в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╦trace_0
К
╠trace_02ы
D__inference_edge_ide2_layer_call_and_return_conditional_losses_19103в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╠trace_0
": @`2edge_ide2/kernel
:`2edge_ide2/bias
"
_generic_user_object
.
[0
\1"
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
═non_trainable_variables
╬layers
╧metrics
 ╨layer_regularization_losses
╤layer_metrics
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
√
╥trace_02▄
5__inference_layer_normalization_2_layer_call_fn_19112в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╥trace_0
Ц
╙trace_02ў
P__inference_layer_normalization_2_layer_call_and_return_conditional_losses_19159в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╙trace_0
 "
trackable_list_wrapper
):'`2layer_normalization_2/gamma
(:&`2layer_normalization_2/beta
.
d0
e1"
trackable_list_wrapper
.
d0
e1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
╘non_trainable_variables
╒layers
╓metrics
 ╫layer_regularization_losses
╪layer_metrics
]	variables
^trainable_variables
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
√
┘trace_02▄
5__inference_layer_normalization_3_layer_call_fn_19168в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z┘trace_0
Ц
┌trace_02ў
P__inference_layer_normalization_3_layer_call_and_return_conditional_losses_19215в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z┌trace_0
 "
trackable_list_wrapper
):'`2layer_normalization_3/gamma
(:&`2layer_normalization_3/beta
"
_generic_user_object
Y
y0
z1
{2
|3
}4
~5
6
А7"
trackable_list_wrapper
Y
y0
z1
{2
|3
}4
~5
6
А7"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
█non_trainable_variables
▄layers
▌metrics
 ▐layer_regularization_losses
▀layer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
╟
рtrace_0
сtrace_12М
#__inference_mpn_layer_call_fn_19248
#__inference_mpn_layer_call_fn_19281┐
╢▓▓
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 zрtrace_0zсtrace_1
¤
тtrace_0
уtrace_12┬
>__inference_mpn_layer_call_and_return_conditional_losses_19563
>__inference_mpn_layer_call_and_return_conditional_losses_19845┐
╢▓▓
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 zтtrace_0zуtrace_1
л
ф	variables
хtrainable_variables
цregularization_losses
ч	keras_api
ш__call__
+щ&call_and_return_all_conditional_losses"
_tf_keras_layer
П
ъlayer_with_weights-0
ъlayer-0
layer-1
ыlayer_with_weights-1
ыlayer-2
ь	variables
эtrainable_variables
юregularization_losses
я	keras_api
Ё__call__
+ё&call_and_return_all_conditional_losses"
_tf_keras_sequential
┴
Є	variables
єtrainable_variables
Їregularization_losses
ї	keras_api
Ў__call__
+ў&call_and_return_all_conditional_losses

}kernel
~bias"
_tf_keras_layer
ц
layer-0
°layer_with_weights-0
°layer-1
∙	variables
·trainable_variables
√regularization_losses
№	keras_api
¤__call__
+■&call_and_return_all_conditional_losses"
_tf_keras_sequential
.
w0
x1"
trackable_list_wrapper
.
w0
x1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
 non_trainable_variables
Аlayers
Бmetrics
 Вlayer_regularization_losses
Гlayer_metrics
q	variables
rtrainable_variables
sregularization_losses
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses"
_generic_user_object
ї
Дtrace_02╓
/__inference_node_prediction_layer_call_fn_19854в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zДtrace_0
Р
Еtrace_02ё
J__inference_node_prediction_layer_call_and_return_conditional_losses_19884в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЕtrace_0
(:&`2node_prediction/kernel
": 2node_prediction/bias
:	а`2dense/kernel
:`2
dense/bias
):'`2layer_normalization_4/gamma
(:&`2layer_normalization_4/beta
%:#	└`2mpn/dense_1/kernel
:`2mpn/dense_1/bias
):'`2layer_normalization_5/gamma
(:&`2layer_normalization_5/beta
 "
trackable_list_wrapper
Ю
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16"
trackable_list_wrapper
0
Ж0
З1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ТBП
%__inference_model_layer_call_fn_16468input_1input_2input_3input_4"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЦBУ
%__inference_model_layer_call_fn_17471inputs_0inputs_1inputs_2inputs_3"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЦBУ
%__inference_model_layer_call_fn_17531inputs_0inputs_1inputs_2inputs_3"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ТBП
%__inference_model_layer_call_fn_17179input_1input_2input_3input_4"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
▒Bо
@__inference_model_layer_call_and_return_conditional_losses_18149inputs_0inputs_1inputs_2inputs_3"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
▒Bо
@__inference_model_layer_call_and_return_conditional_losses_18767inputs_0inputs_1inputs_2inputs_3"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
нBк
@__inference_model_layer_call_and_return_conditional_losses_17263input_1input_2input_3input_4"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
нBк
@__inference_model_layer_call_and_return_conditional_losses_17347input_1input_2input_3input_4"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
є
П0
И1
Й2
К3
Л4
М5
Н6
О7
П8
Р9
С10
Т11
У12
Ф13
Х14
Ц15
Ч16
Ш17
Щ18
Ъ19
Ы20
Ь21
Э22
Ю23
Я24
а25
б26
в27
г28
д29
е30
ж31
з32
и33
й34
к35
л36
м37
н38
о39
п40
░41
▒42
▓43
│44
┤45
╡46
╢47
╖48
╕49
╣50
║51
╗52"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
А
И0
К1
М2
О3
Р4
Т5
Ф6
Ц7
Ш8
Ъ9
Ь10
Ю11
а12
в13
д14
ж15
и16
к17
м18
о19
░20
▓21
┤22
╢23
╕24
║25"
trackable_list_wrapper
А
Й0
Л1
Н2
П3
С4
У5
Х6
Ч7
Щ8
Ы9
Э10
Я11
б12
г13
е14
з15
й16
л17
н18
п19
▒20
│21
╡22
╖23
╣24
╗25"
trackable_list_wrapper
┐2╝╣
о▓к
FullArgSpec2
args*Ъ'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
уBр
#__inference_signature_wrapper_17411input_1input_2input_3input_4"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
▌B┌
)__inference_node_ide1_layer_call_fn_18776inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
°Bї
D__inference_node_ide1_layer_call_and_return_conditional_losses_18806inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ўBЇ
&__inference_lambda_layer_call_fn_18811inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsЪ

 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ўBЇ
&__inference_lambda_layer_call_fn_18816inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsЪ

 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ўBЇ
&__inference_lambda_layer_call_fn_18821inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsЪ

 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ўBЇ
&__inference_lambda_layer_call_fn_18826inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsЪ

 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ТBП
A__inference_lambda_layer_call_and_return_conditional_losses_18838inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsЪ

 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ТBП
A__inference_lambda_layer_call_and_return_conditional_losses_18850inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsЪ

 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ТBП
A__inference_lambda_layer_call_and_return_conditional_losses_18862inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsЪ

 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ТBП
A__inference_lambda_layer_call_and_return_conditional_losses_18874inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsЪ

 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
чBф
3__inference_layer_normalization_layer_call_fn_18883inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ВB 
N__inference_layer_normalization_layer_call_and_return_conditional_losses_18930inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
▌B┌
)__inference_node_ide2_layer_call_fn_18939inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
°Bї
D__inference_node_ide2_layer_call_and_return_conditional_losses_18969inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
▌B┌
)__inference_edge_ide1_layer_call_fn_18978inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
°Bї
D__inference_edge_ide1_layer_call_and_return_conditional_losses_19008inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
щBц
5__inference_layer_normalization_1_layer_call_fn_19017inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ДBБ
P__inference_layer_normalization_1_layer_call_and_return_conditional_losses_19064inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
▌B┌
)__inference_edge_ide2_layer_call_fn_19073inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
°Bї
D__inference_edge_ide2_layer_call_and_return_conditional_losses_19103inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
щBц
5__inference_layer_normalization_2_layer_call_fn_19112inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ДBБ
P__inference_layer_normalization_2_layer_call_and_return_conditional_losses_19159inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
щBц
5__inference_layer_normalization_3_layer_call_fn_19168inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ДBБ
P__inference_layer_normalization_3_layer_call_and_return_conditional_losses_19215inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
<
m0
n1
o2
p3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЮBЫ
#__inference_mpn_layer_call_fn_19248inputs_0inputs_1inputs_2inputs_3inputs_4"┐
╢▓▓
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
ЮBЫ
#__inference_mpn_layer_call_fn_19281inputs_0inputs_1inputs_2inputs_3inputs_4"┐
╢▓▓
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
╣B╢
>__inference_mpn_layer_call_and_return_conditional_losses_19563inputs_0inputs_1inputs_2inputs_3inputs_4"┐
╢▓▓
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
╣B╢
>__inference_mpn_layer_call_and_return_conditional_losses_19845inputs_0inputs_1inputs_2inputs_3inputs_4"┐
╢▓▓
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╝non_trainable_variables
╜layers
╛metrics
 ┐layer_regularization_losses
└layer_metrics
ф	variables
хtrainable_variables
цregularization_losses
ш__call__
+щ&call_and_return_all_conditional_losses
'щ"call_and_return_conditional_losses"
_generic_user_object
┼2┬┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsЪ

 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
┼2┬┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsЪ

 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
┴
┴	variables
┬trainable_variables
├regularization_losses
─	keras_api
┼__call__
+╞&call_and_return_all_conditional_losses

ykernel
zbias"
_tf_keras_layer
╦
╟	variables
╚trainable_variables
╔regularization_losses
╩	keras_api
╦__call__
+╠&call_and_return_all_conditional_losses
	═axis
	{gamma
|beta"
_tf_keras_layer
<
y0
z1
{2
|3"
trackable_list_wrapper
<
y0
z1
{2
|3"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╬non_trainable_variables
╧layers
╨metrics
 ╤layer_regularization_losses
╥layer_metrics
ь	variables
эtrainable_variables
юregularization_losses
Ё__call__
+ё&call_and_return_all_conditional_losses
'ё"call_and_return_conditional_losses"
_generic_user_object
х
╙trace_0
╘trace_1
╒trace_2
╓trace_32Є
*__inference_sequential_layer_call_fn_15431
*__inference_sequential_layer_call_fn_19897
*__inference_sequential_layer_call_fn_19910
*__inference_sequential_layer_call_fn_15528┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╙trace_0z╘trace_1z╒trace_2z╓trace_3
╤
╫trace_0
╪trace_1
┘trace_2
┌trace_32▐
E__inference_sequential_layer_call_and_return_conditional_losses_19991
E__inference_sequential_layer_call_and_return_conditional_losses_20072
E__inference_sequential_layer_call_and_return_conditional_losses_15543
E__inference_sequential_layer_call_and_return_conditional_losses_15558┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╫trace_0z╪trace_1z┘trace_2z┌trace_3
.
}0
~1"
trackable_list_wrapper
.
}0
~1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
█non_trainable_variables
▄layers
▌metrics
 ▐layer_regularization_losses
▀layer_metrics
Є	variables
єtrainable_variables
Їregularization_losses
Ў__call__
+ў&call_and_return_all_conditional_losses
'ў"call_and_return_conditional_losses"
_generic_user_object
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╠
р	variables
сtrainable_variables
тregularization_losses
у	keras_api
ф__call__
+х&call_and_return_all_conditional_losses
	цaxis
	gamma
	Аbeta"
_tf_keras_layer
/
0
А1"
trackable_list_wrapper
/
0
А1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
чnon_trainable_variables
шlayers
щmetrics
 ъlayer_regularization_losses
ыlayer_metrics
∙	variables
·trainable_variables
√regularization_losses
¤__call__
+■&call_and_return_all_conditional_losses
'■"call_and_return_conditional_losses"
_generic_user_object
э
ьtrace_0
эtrace_1
юtrace_2
яtrace_32·
,__inference_sequential_1_layer_call_fn_15627
,__inference_sequential_1_layer_call_fn_20081
,__inference_sequential_1_layer_call_fn_20090
,__inference_sequential_1_layer_call_fn_15674┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zьtrace_0zэtrace_1zюtrace_2zяtrace_3
┘
Ёtrace_0
ёtrace_1
Єtrace_2
єtrace_32ц
G__inference_sequential_1_layer_call_and_return_conditional_losses_20145
G__inference_sequential_1_layer_call_and_return_conditional_losses_20200
G__inference_sequential_1_layer_call_and_return_conditional_losses_15684
G__inference_sequential_1_layer_call_and_return_conditional_losses_15694┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЁtrace_0zёtrace_1zЄtrace_2zєtrace_3
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
уBр
/__inference_node_prediction_layer_call_fn_19854inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
■B√
J__inference_node_prediction_layer_call_and_return_conditional_losses_19884inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
R
Ї	variables
ї	keras_api

Ўtotal

ўcount"
_tf_keras_metric
c
°	variables
∙	keras_api

·total

√count
№
_fn_kwargs"
_tf_keras_metric
':%@2Adam/m/node_ide1/kernel
':%@2Adam/v/node_ide1/kernel
!:@2Adam/m/node_ide1/bias
!:@2Adam/v/node_ide1/bias
,:*@2 Adam/m/layer_normalization/gamma
,:*@2 Adam/v/layer_normalization/gamma
+:)@2Adam/m/layer_normalization/beta
+:)@2Adam/v/layer_normalization/beta
':%@`2Adam/m/node_ide2/kernel
':%@`2Adam/v/node_ide2/kernel
!:`2Adam/m/node_ide2/bias
!:`2Adam/v/node_ide2/bias
':%@2Adam/m/edge_ide1/kernel
':%@2Adam/v/edge_ide1/kernel
!:@2Adam/m/edge_ide1/bias
!:@2Adam/v/edge_ide1/bias
.:,@2"Adam/m/layer_normalization_1/gamma
.:,@2"Adam/v/layer_normalization_1/gamma
-:+@2!Adam/m/layer_normalization_1/beta
-:+@2!Adam/v/layer_normalization_1/beta
':%@`2Adam/m/edge_ide2/kernel
':%@`2Adam/v/edge_ide2/kernel
!:`2Adam/m/edge_ide2/bias
!:`2Adam/v/edge_ide2/bias
.:,`2"Adam/m/layer_normalization_2/gamma
.:,`2"Adam/v/layer_normalization_2/gamma
-:+`2!Adam/m/layer_normalization_2/beta
-:+`2!Adam/v/layer_normalization_2/beta
.:,`2"Adam/m/layer_normalization_3/gamma
.:,`2"Adam/v/layer_normalization_3/gamma
-:+`2!Adam/m/layer_normalization_3/beta
-:+`2!Adam/v/layer_normalization_3/beta
$:"	а`2Adam/m/dense/kernel
$:"	а`2Adam/v/dense/kernel
:`2Adam/m/dense/bias
:`2Adam/v/dense/bias
.:,`2"Adam/m/layer_normalization_4/gamma
.:,`2"Adam/v/layer_normalization_4/gamma
-:+`2!Adam/m/layer_normalization_4/beta
-:+`2!Adam/v/layer_normalization_4/beta
*:(	└`2Adam/m/mpn/dense_1/kernel
*:(	└`2Adam/v/mpn/dense_1/kernel
#:!`2Adam/m/mpn/dense_1/bias
#:!`2Adam/v/mpn/dense_1/bias
.:,`2"Adam/m/layer_normalization_5/gamma
.:,`2"Adam/v/layer_normalization_5/gamma
-:+`2!Adam/m/layer_normalization_5/beta
-:+`2!Adam/v/layer_normalization_5/beta
-:+`2Adam/m/node_prediction/kernel
-:+`2Adam/v/node_prediction/kernel
':%2Adam/m/node_prediction/bias
':%2Adam/v/node_prediction/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
y0
z1"
trackable_list_wrapper
.
y0
z1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
¤non_trainable_variables
■layers
 metrics
 Аlayer_regularization_losses
Бlayer_metrics
┴	variables
┬trainable_variables
├regularization_losses
┼__call__
+╞&call_and_return_all_conditional_losses
'╞"call_and_return_conditional_losses"
_generic_user_object
ы
Вtrace_02╠
%__inference_dense_layer_call_fn_20209в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zВtrace_0
Ж
Гtrace_02ч
@__inference_dense_layer_call_and_return_conditional_losses_20239в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zГtrace_0
.
{0
|1"
trackable_list_wrapper
.
{0
|1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Дnon_trainable_variables
Еlayers
Жmetrics
 Зlayer_regularization_losses
Иlayer_metrics
╟	variables
╚trainable_variables
╔regularization_losses
╦__call__
+╠&call_and_return_all_conditional_losses
'╠"call_and_return_conditional_losses"
_generic_user_object
√
Йtrace_02▄
5__inference_layer_normalization_4_layer_call_fn_20248в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЙtrace_0
Ц
Кtrace_02ў
P__inference_layer_normalization_4_layer_call_and_return_conditional_losses_20295в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zКtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
7
ъ0
1
ы2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
АB¤
*__inference_sequential_layer_call_fn_15431dense_input"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
√B°
*__inference_sequential_layer_call_fn_19897inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
√B°
*__inference_sequential_layer_call_fn_19910inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
АB¤
*__inference_sequential_layer_call_fn_15528dense_input"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЦBУ
E__inference_sequential_layer_call_and_return_conditional_losses_19991inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЦBУ
E__inference_sequential_layer_call_and_return_conditional_losses_20072inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЫBШ
E__inference_sequential_layer_call_and_return_conditional_losses_15543dense_input"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЫBШ
E__inference_sequential_layer_call_and_return_conditional_losses_15558dense_input"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
/
0
А1"
trackable_list_wrapper
/
0
А1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Лnon_trainable_variables
Мlayers
Нmetrics
 Оlayer_regularization_losses
Пlayer_metrics
р	variables
сtrainable_variables
тregularization_losses
ф__call__
+х&call_and_return_all_conditional_losses
'х"call_and_return_conditional_losses"
_generic_user_object
√
Рtrace_02▄
5__inference_layer_normalization_5_layer_call_fn_20304в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zРtrace_0
Ц
Сtrace_02ў
P__inference_layer_normalization_5_layer_call_and_return_conditional_losses_20351в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zСtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
/
0
°1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ГBА
,__inference_sequential_1_layer_call_fn_15627lambda_input"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
¤B·
,__inference_sequential_1_layer_call_fn_20081inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
¤B·
,__inference_sequential_1_layer_call_fn_20090inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ГBА
,__inference_sequential_1_layer_call_fn_15674lambda_input"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ШBХ
G__inference_sequential_1_layer_call_and_return_conditional_losses_20145inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ШBХ
G__inference_sequential_1_layer_call_and_return_conditional_losses_20200inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЮBЫ
G__inference_sequential_1_layer_call_and_return_conditional_losses_15684lambda_input"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЮBЫ
G__inference_sequential_1_layer_call_and_return_conditional_losses_15694lambda_input"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
0
Ў0
ў1"
trackable_list_wrapper
.
Ї	variables"
_generic_user_object
:  (2total
:  (2count
0
·0
√1"
trackable_list_wrapper
.
°	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
┘B╓
%__inference_dense_layer_call_fn_20209inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЇBё
@__inference_dense_layer_call_and_return_conditional_losses_20239inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
щBц
5__inference_layer_normalization_4_layer_call_fn_20248inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ДBБ
P__inference_layer_normalization_4_layer_call_and_return_conditional_losses_20295inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
щBц
5__inference_layer_normalization_5_layer_call_fn_20304inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ДBБ
P__inference_layer_normalization_5_layer_call_and_return_conditional_losses_20351inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 ь
 __inference__wrapped_model_15309╟!"01@A89IJQR[\deyz{|}~Аwx╫в╙
╦в╟
─Ъ└
.К+
input_1                  
.К+
input_2                  
.К+
input_3                  
.К+
input_4                  
к "NкK
I
node_prediction6К3
node_prediction                  ┬
@__inference_dense_layer_call_and_return_conditional_losses_20239~yz=в:
3в0
.К+
inputs                  а
к "9в6
/К,
tensor_0                  `
Ъ Ь
%__inference_dense_layer_call_fn_20209syz=в:
3в0
.К+
inputs                  а
к ".К+
unknown                  `┼
D__inference_edge_ide1_layer_call_and_return_conditional_losses_19008}@A<в9
2в/
-К*
inputs                  
к "9в6
/К,
tensor_0                  @
Ъ Я
)__inference_edge_ide1_layer_call_fn_18978r@A<в9
2в/
-К*
inputs                  
к ".К+
unknown                  @┼
D__inference_edge_ide2_layer_call_and_return_conditional_losses_19103}QR<в9
2в/
-К*
inputs                  @
к "9в6
/К,
tensor_0                  `
Ъ Я
)__inference_edge_ide2_layer_call_fn_19073rQR<в9
2в/
-К*
inputs                  @
к ".К+
unknown                  `╟
A__inference_lambda_layer_call_and_return_conditional_losses_18838БDвA
:в7
-К*
inputs                  `

 
p 
к "9в6
/К,
tensor_0                  `
Ъ ╟
A__inference_lambda_layer_call_and_return_conditional_losses_18850БDвA
:в7
-К*
inputs                  `

 
p
к "9в6
/К,
tensor_0                  `
Ъ ╟
A__inference_lambda_layer_call_and_return_conditional_losses_18862БDвA
:в7
-К*
inputs                  @

 
p 
к "9в6
/К,
tensor_0                  @
Ъ ╟
A__inference_lambda_layer_call_and_return_conditional_losses_18874БDвA
:в7
-К*
inputs                  @

 
p
к "9в6
/К,
tensor_0                  @
Ъ а
&__inference_lambda_layer_call_fn_18811vDвA
:в7
-К*
inputs                  `

 
p 
к ".К+
unknown                  `а
&__inference_lambda_layer_call_fn_18816vDвA
:в7
-К*
inputs                  `

 
p
к ".К+
unknown                  `а
&__inference_lambda_layer_call_fn_18821vDвA
:в7
-К*
inputs                  @

 
p 
к ".К+
unknown                  @а
&__inference_lambda_layer_call_fn_18826vDвA
:в7
-К*
inputs                  @

 
p
к ".К+
unknown                  @╤
P__inference_layer_normalization_1_layer_call_and_return_conditional_losses_19064}IJ<в9
2в/
-К*
inputs                  @
к "9в6
/К,
tensor_0                  @
Ъ л
5__inference_layer_normalization_1_layer_call_fn_19017rIJ<в9
2в/
-К*
inputs                  @
к ".К+
unknown                  @╤
P__inference_layer_normalization_2_layer_call_and_return_conditional_losses_19159}[\<в9
2в/
-К*
inputs                  `
к "9в6
/К,
tensor_0                  `
Ъ л
5__inference_layer_normalization_2_layer_call_fn_19112r[\<в9
2в/
-К*
inputs                  `
к ".К+
unknown                  `╤
P__inference_layer_normalization_3_layer_call_and_return_conditional_losses_19215}de<в9
2в/
-К*
inputs                  `
к "9в6
/К,
tensor_0                  `
Ъ л
5__inference_layer_normalization_3_layer_call_fn_19168rde<в9
2в/
-К*
inputs                  `
к ".К+
unknown                  `╤
P__inference_layer_normalization_4_layer_call_and_return_conditional_losses_20295}{|<в9
2в/
-К*
inputs                  `
к "9в6
/К,
tensor_0                  `
Ъ л
5__inference_layer_normalization_4_layer_call_fn_20248r{|<в9
2в/
-К*
inputs                  `
к ".К+
unknown                  `╥
P__inference_layer_normalization_5_layer_call_and_return_conditional_losses_20351~А<в9
2в/
-К*
inputs                  `
к "9в6
/К,
tensor_0                  `
Ъ м
5__inference_layer_normalization_5_layer_call_fn_20304sА<в9
2в/
-К*
inputs                  `
к ".К+
unknown                  `╧
N__inference_layer_normalization_layer_call_and_return_conditional_losses_18930}01<в9
2в/
-К*
inputs                  @
к "9в6
/К,
tensor_0                  @
Ъ й
3__inference_layer_normalization_layer_call_fn_18883r01<в9
2в/
-К*
inputs                  @
к ".К+
unknown                  @ 
@__inference_model_layer_call_and_return_conditional_losses_17263║!"01@A89IJQR[\deyz{|}~Аwx▀в█
╙в╧
─Ъ└
.К+
input_1                  
.К+
input_2                  
.К+
input_3                  
.К+
input_4                  
p 

 
к "9в6
/К,
tensor_0                  
Ъ  
@__inference_model_layer_call_and_return_conditional_losses_17347║!"01@A89IJQR[\deyz{|}~Аwx▀в█
╙в╧
─Ъ└
.К+
input_1                  
.К+
input_2                  
.К+
input_3                  
.К+
input_4                  
p

 
к "9в6
/К,
tensor_0                  
Ъ Г
@__inference_model_layer_call_and_return_conditional_losses_18149╛!"01@A89IJQR[\deyz{|}~Аwxув▀
╫в╙
╚Ъ─
/К,
inputs_0                  
/К,
inputs_1                  
/К,
inputs_2                  
/К,
inputs_3                  
p 

 
к "9в6
/К,
tensor_0                  
Ъ Г
@__inference_model_layer_call_and_return_conditional_losses_18767╛!"01@A89IJQR[\deyz{|}~Аwxув▀
╫в╙
╚Ъ─
/К,
inputs_0                  
/К,
inputs_1                  
/К,
inputs_2                  
/К,
inputs_3                  
p

 
к "9в6
/К,
tensor_0                  
Ъ ┘
%__inference_model_layer_call_fn_16468п!"01@A89IJQR[\deyz{|}~Аwx▀в█
╙в╧
─Ъ└
.К+
input_1                  
.К+
input_2                  
.К+
input_3                  
.К+
input_4                  
p 

 
к ".К+
unknown                  ┘
%__inference_model_layer_call_fn_17179п!"01@A89IJQR[\deyz{|}~Аwx▀в█
╙в╧
─Ъ└
.К+
input_1                  
.К+
input_2                  
.К+
input_3                  
.К+
input_4                  
p

 
к ".К+
unknown                  ▌
%__inference_model_layer_call_fn_17471│!"01@A89IJQR[\deyz{|}~Аwxув▀
╫в╙
╚Ъ─
/К,
inputs_0                  
/К,
inputs_1                  
/К,
inputs_2                  
/К,
inputs_3                  
p 

 
к ".К+
unknown                  ▌
%__inference_model_layer_call_fn_17531│!"01@A89IJQR[\deyz{|}~Аwxув▀
╫в╙
╚Ъ─
/К,
inputs_0                  
/К,
inputs_1                  
/К,
inputs_2                  
/К,
inputs_3                  
p

 
к ".К+
unknown                   
>__inference_mpn_layer_call_and_return_conditional_losses_19563╝	yz{|}~АЬвШ
Ав№
∙вї
/К,
inputs_0                  `
/К,
inputs_1                  `
/К,
inputs_2                  
/К,
inputs_3                  
/К,
inputs_4                  
к

trainingp "ПвЛ
Гв 
1К.

tensor_0_0                  `
1К.

tensor_0_1                  `
1К.

tensor_0_2                  
1К.

tensor_0_3                  
1К.

tensor_0_4                  
Ъ  
>__inference_mpn_layer_call_and_return_conditional_losses_19845╝	yz{|}~АЬвШ
Ав№
∙вї
/К,
inputs_0                  `
/К,
inputs_1                  `
/К,
inputs_2                  
/К,
inputs_3                  
/К,
inputs_4                  
к

trainingp"ПвЛ
Гв 
1К.

tensor_0_0                  `
1К.

tensor_0_1                  `
1К.

tensor_0_2                  
1К.

tensor_0_3                  
1К.

tensor_0_4                  
Ъ ╬
#__inference_mpn_layer_call_fn_19248ж	yz{|}~АЬвШ
Ав№
∙вї
/К,
inputs_0                  `
/К,
inputs_1                  `
/К,
inputs_2                  
/К,
inputs_3                  
/К,
inputs_4                  
к

trainingp "∙вї
/К,
tensor_0                  `
/К,
tensor_1                  `
/К,
tensor_2                  
/К,
tensor_3                  
/К,
tensor_4                  ╬
#__inference_mpn_layer_call_fn_19281ж	yz{|}~АЬвШ
Ав№
∙вї
/К,
inputs_0                  `
/К,
inputs_1                  `
/К,
inputs_2                  
/К,
inputs_3                  
/К,
inputs_4                  
к

trainingp"∙вї
/К,
tensor_0                  `
/К,
tensor_1                  `
/К,
tensor_2                  
/К,
tensor_3                  
/К,
tensor_4                  ┼
D__inference_node_ide1_layer_call_and_return_conditional_losses_18806}!"<в9
2в/
-К*
inputs                  
к "9в6
/К,
tensor_0                  @
Ъ Я
)__inference_node_ide1_layer_call_fn_18776r!"<в9
2в/
-К*
inputs                  
к ".К+
unknown                  @┼
D__inference_node_ide2_layer_call_and_return_conditional_losses_18969}89<в9
2в/
-К*
inputs                  @
к "9в6
/К,
tensor_0                  `
Ъ Я
)__inference_node_ide2_layer_call_fn_18939r89<в9
2в/
-К*
inputs                  @
к ".К+
unknown                  `╦
J__inference_node_prediction_layer_call_and_return_conditional_losses_19884}wx<в9
2в/
-К*
inputs                  `
к "9в6
/К,
tensor_0                  
Ъ е
/__inference_node_prediction_layer_call_fn_19854rwx<в9
2в/
-К*
inputs                  `
к ".К+
unknown                  ╪
G__inference_sequential_1_layer_call_and_return_conditional_losses_15684МАJвG
@в=
3К0
lambda_input                  `
p 

 
к "9в6
/К,
tensor_0                  `
Ъ ╪
G__inference_sequential_1_layer_call_and_return_conditional_losses_15694МАJвG
@в=
3К0
lambda_input                  `
p

 
к "9в6
/К,
tensor_0                  `
Ъ ╥
G__inference_sequential_1_layer_call_and_return_conditional_losses_20145ЖАDвA
:в7
-К*
inputs                  `
p 

 
к "9в6
/К,
tensor_0                  `
Ъ ╥
G__inference_sequential_1_layer_call_and_return_conditional_losses_20200ЖАDвA
:в7
-К*
inputs                  `
p

 
к "9в6
/К,
tensor_0                  `
Ъ ▓
,__inference_sequential_1_layer_call_fn_15627БАJвG
@в=
3К0
lambda_input                  `
p 

 
к ".К+
unknown                  `▓
,__inference_sequential_1_layer_call_fn_15674БАJвG
@в=
3К0
lambda_input                  `
p

 
к ".К+
unknown                  `л
,__inference_sequential_1_layer_call_fn_20081{АDвA
:в7
-К*
inputs                  `
p 

 
к ".К+
unknown                  `л
,__inference_sequential_1_layer_call_fn_20090{АDвA
:в7
-К*
inputs                  `
p

 
к ".К+
unknown                  `╫
E__inference_sequential_layer_call_and_return_conditional_losses_15543Нyz{|JвG
@в=
3К0
dense_input                  а
p 

 
к "9в6
/К,
tensor_0                  `
Ъ ╫
E__inference_sequential_layer_call_and_return_conditional_losses_15558Нyz{|JвG
@в=
3К0
dense_input                  а
p

 
к "9в6
/К,
tensor_0                  `
Ъ ╥
E__inference_sequential_layer_call_and_return_conditional_losses_19991Иyz{|EвB
;в8
.К+
inputs                  а
p 

 
к "9в6
/К,
tensor_0                  `
Ъ ╥
E__inference_sequential_layer_call_and_return_conditional_losses_20072Иyz{|EвB
;в8
.К+
inputs                  а
p

 
к "9в6
/К,
tensor_0                  `
Ъ ▒
*__inference_sequential_layer_call_fn_15431Вyz{|JвG
@в=
3К0
dense_input                  а
p 

 
к ".К+
unknown                  `▒
*__inference_sequential_layer_call_fn_15528Вyz{|JвG
@в=
3К0
dense_input                  а
p

 
к ".К+
unknown                  `л
*__inference_sequential_layer_call_fn_19897}yz{|EвB
;в8
.К+
inputs                  а
p 

 
к ".К+
unknown                  `л
*__inference_sequential_layer_call_fn_19910}yz{|EвB
;в8
.К+
inputs                  а
p

 
к ".К+
unknown                  `Ф
#__inference_signature_wrapper_17411ь!"01@A89IJQR[\deyz{|}~Аwx№в°
в 
Ёкь
9
input_1.К+
input_1                  
9
input_2.К+
input_2                  
9
input_3.К+
input_3                  
9
input_4.К+
input_4                  "NкK
I
node_prediction6К3
node_prediction                  