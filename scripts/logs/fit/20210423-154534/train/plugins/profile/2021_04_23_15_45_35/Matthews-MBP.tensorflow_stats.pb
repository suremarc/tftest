"?]
BHostIDLE"IDLE1    @R?@A    @R?@a??b?????i??b??????Unknown
?HostConv2DBackpropFilter":gradient_tape/sequential/conv0/Conv2D/Conv2DBackpropFilter(1     |?@9     |?@A     |?@I     |?@a??d????i????g????Unknown
?HostBiasAddGrad"2gradient_tape/sequential/conv0/BiasAdd/BiasAddGrad(1     T?@9     T?@A     T?@I     T?@a??????i{P9u???Unknown
?HostMaxPoolGrad"2gradient_tape/sequential/pool0/MaxPool/MaxPoolGrad(1     ?@9     ?@A     ?@I     ?@a???X?U??i?Z?
???Unknown
oHost_FusedConv2D"sequential/conv0/Relu(1     \?@9     \?@A     \?@I     \?@a?ʩC????is?=j?#???Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1      z@9      z@A      z@I      z@a?+S)u??i"D??????Unknown
uHostMatMul"!sequential/dense/Tensordot/MatMul(1     @y@9     @y@A     @y@I     @y@a?=?N???i}gJ?????Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1     ?s@9     ?s@A     ?s@I     ?s@aB?8a ??i!:K??C???Unknown
q	HostMul" sequential/dropout_1/dropout/Mul(1     `s@9     `s@A     `s@I     `s@aP%?I(???i??qpw????Unknown
?
HostRandomUniform"9sequential/dropout_1/dropout/random_uniform/RandomUniform(1     Pr@9     Pr@A     Pr@I     Pr@a)ǣ????iZ7?????Unknown
mHostMaxPool"sequential/pool0/MaxPool(1     ?p@9     ?p@A     ?p@I     ?p@as?o1ր?i????????Unknown
}HostReluGrad"'gradient_tape/sequential/conv0/ReluGrad(1     ?m@9     ?m@A     ?m@I     ?m@ao?'%? ~?i'??@\???Unknown
^HostGatherV2"GatherV2(1     `h@9     `h@A     `h@I     `h@a?Oɿt?x?i?c?ڟ????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_4/ResourceApplyAdam(1     `g@9     `g@A     `g@I     `g@aح?A2?w?i#?
??????Unknown
?HostMatMul"8gradient_tape/sequential/dense/Tensordot/MatMul/MatMul_1(1     ?b@9     ?b@A     ?b@I     ?b@a??(k??r?i?<᫱????Unknown
tHost_FusedMatMul"sequential/dense_1/BiasAdd(1      b@9      b@A      b@I      b@abfeܬ:r?i??'???Unknown
?HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1     ?`@9     ?`@A     ?`@I     ?`@as?o1?p?iCxh?(???Unknown
?HostRandomUniform"7sequential/dropout/dropout/random_uniform/RandomUniform(1     ``@9     ``@A     ``@I     ``@a?>??`?p?i?{*?I???Unknown
?HostMatMul"6gradient_tape/sequential/dense/Tensordot/MatMul/MatMul(1      `@9      `@A      `@I      `@a"Z?'4p?i?/?yfj???Unknown
?HostGreaterEqual")sequential/dropout_1/dropout/GreaterEqual(1     ?_@9     ?_@A     ?_@I     ?_@a+s????o?iK?Y(M????Unknown
}HostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1     @]@9     @]@A     @]@I     @]@a_?$?X?m?i@??????Unknown
?HostMul"0gradient_tape/sequential/dropout_1/dropout/Mul_1(1      Y@9      Y@A      Y@I      Y@aOՌN~Qi?i撎?=????Unknown
sHostMul""sequential/dropout_1/dropout/Mul_1(1     ?W@9     ?W@A     ?W@I     ?W@a??1kh?i???jK????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1     @W@9     @W@A     @W@I     @W@a???ɋg?i???4?????Unknown?
dHostDataset"Iterator::Model(1     @V@9     @V@A     @V@I     @V@aqW?s??f?i?%?_???Unknown
?HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1     ?U@9     ?U@A     ?U@I     ?U@aa??4?f?i??Z?f???Unknown
?HostGreaterEqual"'sequential/dropout/dropout/GreaterEqual(1      S@9      S@A      S@I      S@a?kZ?=c?i?????0???Unknown
gHostRelu"sequential/dense/Relu(1     ?R@9     ?R@A     ?R@I     ?R@a?????b?i1?o??C???Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1     @Q@9     @Q@A     ?O@I     ?O@a+s????_?i????S???Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1     ?N@9     ?N@A     ?L@I     ?L@aŌ???\?i1?4{b???Unknown
mHostBiasAdd"sequential/dense/BiasAdd(1     ?L@9     ?L@A     ?L@I     ?L@aŌ???\?iwa??qp???Unknown
? HostSoftmaxCrossEntropyWithLogits":categorical_crossentropy/softmax_cross_entropy_with_logits(1     ?F@9     ?F@A     ?F@I     ?F@a??~X?V?i? ?{???Unknown
q!HostMul" sequential/dropout/dropout/Mul_1(1     ?C@9     ?C@A     ?C@I     ?C@a??m???S?i??c?????Unknown
s"HostCast"!sequential/dropout_1/dropout/Cast(1     ?C@9     ?C@A     ?C@I     ?C@a??m???S?i??[+?????Unknown
t#HostReadVariableOp"Adam/Cast/ReadVariableOp(1     ?A@9     ?A@A     ?A@I     ?A@aQ?b??Q?i??*?r????Unknown
o$HostSoftmax"sequential/dense_1/Softmax(1      =@9      =@A      =@I      =@a?]?F?^M?i??;Sʟ???Unknown
?%HostResourceApplyAdam"$Adam/Adam/update_5/ResourceApplyAdam(1      ;@9      ;@A      ;@I      ;@a??JXK?iَT?????Unknown
V&HostSum"Sum_2(1      ;@9      ;@A      ;@I      ;@a??JXK?i?4?Tv????Unknown
g'HostStridedSlice"strided_slice(1      9@9      9@A      9@I      9@aOՌN~QI?i?t?ʳ???Unknown
{(HostGatherV2"%sequential/dense/Tensordot/GatherV2_1(1      6@9      6@A      6@I      6@a??{ԶGF?i?)?\????Unknown
)HostMul".gradient_tape/sequential/dropout_1/dropout/Mul(1      5@9      5@A      5@I      5@a?LvVtDE?i?????????Unknown
q*HostProd"sequential/dense/Tensordot/Prod(1      5@9      5@A      5@I      5@a?LvVtDE?i62U??????Unknown
?+HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1      3@9      3@A      3@I      3@a?kZ?=C?i??+X?????Unknown
l,HostIteratorGetNext"IteratorGetNext(1      2@9      2@A      2@I      2@abfeܬ:B?iR?b]????Unknown
?-HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1      .@9      .@A      .@I      .@a?????a>?ir{?<)????Unknown
i.HostWriteSummary"WriteSummary(1      .@9      .@A      .@I      .@a?????a>?i?v?????Unknown?
?/HostMul"Lgradient_tape/categorical_crossentropy/softmax_cross_entropy_with_logits/mul(1      *@9      *@A      *@I      *@apw???T:?i??-@????Unknown
?0HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1      "@9      "@A      "@I      "@abfeܬ:2?i?/?c?????Unknown
q1HostCast"sequential/dropout/dropout/Cast(1      "@9      "@A      "@I      "@abfeܬ:2?i;?d??????Unknown
Z2HostArgMax"ArgMax(1       @9       @A       @I       @a"Z?'40?i?`>?????Unknown
`3HostGatherV2"
GatherV2_1(1       @9       @A       @I       @a"Z?'40?i??\??????Unknown
x4HostDataset"#Iterator::Model::ParallelMapV2::Zip(1     ?a@9     ?a@A       @I       @a"Z?'40?i?XH?????Unknown
?5HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @a-3??;N(?iz?,g????Unknown
e6Host
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @a-3??;N(?i????????Unknown?
o7HostMul"sequential/dropout/dropout/Mul(1      @9      @A      @I      @a-3??;N(?i`???p????Unknown
~8HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1      @9      @A      @I      @a??p?1A$?ik~??????Unknown
[9HostAddV2"Adam/add(1      @9      @A      @I      @a??p?1A$?iv??????Unknown
v:HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a??p?1A$?i???,=????Unknown
\;HostArgMax"ArgMax_1(1      @9      @A      @I      @a"Z?'4 ?i#?fo@????Unknown
?<HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9       @A      @I       @a"Z?'4 ?iŗ??C????Unknown
?=HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a"Z?'4 ?ig?b?F????Unknown
y>HostGatherV2"#sequential/dense/Tensordot/GatherV2(1      @9      @A      @I      @a"Z?'4 ?i	??6J????Unknown
??HostReadVariableOp")sequential/dense/Tensordot/ReadVariableOp(1      @9      @A      @I      @a"Z?'4 ?i??^yM????Unknown
y@HostConcatV2"#sequential/dense/Tensordot/concat_1(1      @9      @A      @I      @a"Z?'4 ?iM?ܻP????Unknown
vAHostReadVariableOp"Adam/Cast_3/ReadVariableOp(1      @9      @A      @I      @a-3??;N?i?2?-????Unknown
bBHostDivNoNan"div_no_nan_1(1      @9      @A      @I      @a-3??;N?i?????????Unknown
?CHostTile";gradient_tape/categorical_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a-3??;N?i?:x?????Unknown
?DHostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a-3??;N?i5?V?Z????Unknown
rEHostPack" sequential/dense/Tensordot/stack(1      @9      @A      @I      @a-3??;N?ioC5?????Unknown
vFHostReadVariableOp"Adam/Cast_2/ReadVariableOp(1       @9       @A       @I       @a"Z?'4?i@Ft??????Unknown
YGHostPow"Adam/Pow(1       @9       @A       @I       @a"Z?'4?iI?7 ????Unknown
[HHostPow"
Adam/Pow_1(1       @9       @A       @I       @a"Z?'4?i?K?ء????Unknown
oIHostReadVariableOp"Adam/ReadVariableOp(1       @9       @A       @I       @a"Z?'4?i?N1z#????Unknown
vJHostAssignAddVariableOp"AssignAddVariableOp_4(1       @9       @A       @I       @a"Z?'4?i?Qp?????Unknown
VKHostCast"Cast(1       @9       @A       @I       @a"Z?'4?iUT??&????Unknown
XLHostEqual"Equal(1       @9       @A       @I       @a"Z?'4?i&W?]?????Unknown
?MHostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1     ?Q@9     ?Q@A       @I       @a"Z?'4?i?Y-?)????Unknown
`NHostDivNoNan"
div_no_nan(1       @9       @A       @I       @a"Z?'4?i?\l??????Unknown
uOHostReadVariableOp"div_no_nan/ReadVariableOp(1       @9       @A       @I       @a"Z?'4?i?_?A-????Unknown
wPHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a"Z?'4?ijb???????Unknown
?QHostReadVariableOp"'sequential/conv0/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a"Z?'4?i;e)?0????Unknown
?RHostReadVariableOp"&sequential/conv0/Conv2D/ReadVariableOp(1       @9       @A       @I       @a"Z?'4?ihh%?????Unknown
?SHostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a"Z?'4?i?j??3????Unknown
?THostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1       @9       @A       @I       @a"Z?'4?i?m?g?????Unknown
]UHostCast"Adam/Cast_1(1      ??9      ??A      ??I      ??a"Z?'4 ?i??8?????Unknown
tVHostAssignAddVariableOp"AssignAddVariableOp(1      ??9      ??A      ??I      ??a"Z?'4 ?i?p%	7????Unknown
vWHostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??a"Z?'4 ?i????w????Unknown
vXHostAssignAddVariableOp"AssignAddVariableOp_3(1      ??9      ??A      ??I      ??a"Z?'4 ?iRsd??????Unknown
XYHostCast"Cast_1(1      ??9      ??A      ??I      ??a"Z?'4 ?i??{?????Unknown
XZHostCast"Cast_2(1      ??9      ??A      ??I      ??a"Z?'4 ?i$v?K:????Unknown
a[HostIdentity"Identity(1      ??9      ??A      ??I      ??a"Z?'4 ?i??B{????Unknown?
T\HostMul"Mul(1      ??9      ??A      ??I      ??a"Z?'4 ?i?x???????Unknown
{]HostSum"*categorical_crossentropy/weighted_loss/Sum(1      ??9      ??A      ??I      ??a"Z?'4 ?i_????????Unknown
?^HostCast"8categorical_crossentropy/weighted_loss/num_elements/Cast(1      ??9      ??A      ??I      ??a"Z?'4 ?i?{!?=????Unknown
?_HostDivNoNan",categorical_crossentropy/weighted_loss/value(1      ??9      ??A      ??I      ??a"Z?'4 ?i1??^~????Unknown
w`HostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a"Z?'4 ?i?~`/?????Unknown
saHostProd"!sequential/dense/Tensordot/Prod_1(1      ??9      ??A      ??I      ??a"Z?'4 ?i     ???Unknown
LbHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(i     ???Unknown
ncHostDivNoNan"Egradient_tape/categorical_crossentropy/weighted_loss/value/div_no_nan(i     ???Unknown*?\
?HostConv2DBackpropFilter":gradient_tape/sequential/conv0/Conv2D/Conv2DBackpropFilter(1     |?@9     |?@A     |?@I     |?@ai` ?????ii` ??????Unknown
?HostBiasAddGrad"2gradient_tape/sequential/conv0/BiasAdd/BiasAddGrad(1     T?@9     T?@A     T?@I     T?@a?z?l?w??i??X?????Unknown
?HostMaxPoolGrad"2gradient_tape/sequential/pool0/MaxPool/MaxPoolGrad(1     ?@9     ?@A     ?@I     ?@a?}?%??i??? }????Unknown
oHost_FusedConv2D"sequential/conv0/Relu(1     \?@9     \?@A     \?@I     \?@ae??????i0?S??n???Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1      z@9      z@A      z@I      z@aᮻC	|??i??q=?j???Unknown
uHostMatMul"!sequential/dense/Tensordot/MatMul(1     @y@9     @y@A     @y@I     @y@a???3n??i???C^???Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1     ?s@9     ?s@A     ?s@I     ?s@aF?w?;͗?i?Fuĭ???Unknown
qHostMul" sequential/dropout_1/dropout/Mul(1     `s@9     `s@A     `s@I     `s@a B???Y??i?0?z????Unknown
?	HostRandomUniform"9sequential/dropout_1/dropout/random_uniform/RandomUniform(1     Pr@9     Pr@A     Pr@I     Pr@a<?Q ???iU???????Unknown
m
HostMaxPool"sequential/pool0/MaxPool(1     ?p@9     ?p@A     ?p@I     ?p@aX+	??i?G?P(???Unknown
}HostReluGrad"'gradient_tape/sequential/conv0/ReluGrad(1     ?m@9     ?m@A     ?m@I     ?m@a?N?5???i?????????Unknown
^HostGatherV2"GatherV2(1     `h@9     `h@A     `h@I     `h@a???$!`??i!?5;-???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_4/ResourceApplyAdam(1     `g@9     `g@A     `g@I     `g@a????+??i5g5??????Unknown
?HostMatMul"8gradient_tape/sequential/dense/Tensordot/MatMul/MatMul_1(1     ?b@9     ?b@A     ?b@I     ?b@au?s&r??i??????Unknown
tHost_FusedMatMul"sequential/dense_1/BiasAdd(1      b@9      b@A      b@I      b@a???S???i??7lwN???Unknown
?HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1     ?`@9     ?`@A     ?`@I     ?`@aX+	??i/`???????Unknown
?HostRandomUniform"7sequential/dropout/dropout/random_uniform/RandomUniform(1     ``@9     ``@A     ``@I     ``@a՟????i???Ћ????Unknown
?HostMatMul"6gradient_tape/sequential/dense/Tensordot/MatMul/MatMul(1      `@9      `@A      `@I      `@a?nDJH??i?:???:???Unknown
?HostGreaterEqual")sequential/dropout_1/dropout/GreaterEqual(1     ?_@9     ?_@A     ?_@I     ?_@abS\)???i*?k??????Unknown
}HostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1     @]@9     @]@A     @]@I     @]@a???????i????????Unknown
?HostMul"0gradient_tape/sequential/dropout_1/dropout/Mul_1(1      Y@9      Y@A      Y@I      Y@a??
? ~?ih??[	???Unknown
sHostMul""sequential/dropout_1/dropout/Mul_1(1     ?W@9     ?W@A     ?W@I     ?W@a*Y?=N?|?i݆r?B???Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1     @W@9     @W@A     @W@I     @W@a??o?|?i??]??z???Unknown?
dHostDataset"Iterator::Model(1     @V@9     @V@A     @V@I     @V@a)G??z?i???E????Unknown
?HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1     ?U@9     ?U@A     ?U@I     ?U@a???D6z?i+?"?????Unknown
?HostGreaterEqual"'sequential/dropout/dropout/GreaterEqual(1      S@9      S@A      S@I      S@a??B1??v?i??8?}???Unknown
gHostRelu"sequential/dense/Relu(1     ?R@9     ?R@A     ?R@I     ?R@a??0??v?iqIA?????Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1     @Q@9     @Q@A     ?O@I     ?O@abS\)?r?i???e???Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1     ?N@9     ?N@A     ?L@I     ?L@aL?$b,q?i??W?????Unknown
mHostBiasAdd"sequential/dense/BiasAdd(1     ?L@9     ?L@A     ?L@I     ?L@aL?$b,q?i"?W????Unknown
?HostSoftmaxCrossEntropyWithLogits":categorical_crossentropy/softmax_cross_entropy_with_logits(1     ?F@9     ?F@A     ?F@I     ?F@aB?:p?k?i⼃?t????Unknown
q HostMul" sequential/dropout/dropout/Mul_1(1     ?C@9     ?C@A     ?C@I     ?C@af??g?i #??????Unknown
s!HostCast"!sequential/dropout_1/dropout/Cast(1     ?C@9     ?C@A     ?C@I     ?C@af??g?i???t????Unknown
t"HostReadVariableOp"Adam/Cast/ReadVariableOp(1     ?A@9     ?A@A     ?A@I     ?A@a?\?:e?i{a?
?	???Unknown
o#HostSoftmax"sequential/dense_1/Softmax(1      =@9      =@A      =@I      =@az?N?ya?i5e????Unknown
?$HostResourceApplyAdam"$Adam/Adam/update_5/ResourceApplyAdam(1      ;@9      ;@A      ;@I      ;@a?ټ??D`?i"??J+???Unknown
V%HostSum"Sum_2(1      ;@9      ;@A      ;@I      ;@a?ټ??D`?i??f??;???Unknown
g&HostStridedSlice"strided_slice(1      9@9      9@A      9@I      9@a??
? ^?i?Tl?J???Unknown
{'HostGatherV2"%sequential/dense/Tensordot/GatherV2_1(1      6@9      6@A      6@I      6@a?Of?Z?i?`{??W???Unknown
(HostMul".gradient_tape/sequential/dropout_1/dropout/Mul(1      5@9      5@A      5@I      5@a-o?y?NY?i?H8)?d???Unknown
q)HostProd"sequential/dense/Tensordot/Prod(1      5@9      5@A      5@I      5@a-o?y?NY?i?0??0q???Unknown
?*HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1      3@9      3@A      3@I      3@a??B1??V?iQ???|???Unknown
l+HostIteratorGetNext"IteratorGetNext(1      2@9      2@A      2@I      2@a???S?U?i8P?/|????Unknown
?,HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1      .@9      .@A      .@I      .@a?*'??R?i?c??????Unknown
i-HostWriteSummary"WriteSummary(1      .@9      .@A      .@I      .@a?*'??R?ibwt??????Unknown?
?.HostMul"Lgradient_tape/categorical_crossentropy/softmax_cross_entropy_with_logits/mul(1      *@9      *@A      *@I      *@a??2?xUO?iD?Se????Unknown
?/HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1      "@9      "@A      "@I      "@a???S?E?i
???Ѧ???Unknown
q0HostCast"sequential/dropout/dropout/Cast(1      "@9      "@A      "@I      "@a???S?E?i??f?=????Unknown
Z1HostArgMax"ArgMax(1       @9       @A       @I       @a?nDJHC?i???????Unknown
`2HostGatherV2"
GatherV2_1(1       @9       @A       @I       @a?nDJHC?i??"?????Unknown
x3HostDataset"#Iterator::Model::ParallelMapV2::Zip(1     ?a@9     ?a@A       @I       @a?nDJHC?i?5?????Unknown
?4HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @aX?fo?<?i(??Q????Unknown
e5Host
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @aX?fo?<?iʽ?P?????Unknown?
o6HostMul"sequential/dropout/dropout/Mul(1      @9      @A      @I      @aX?fo?<?il??ތ????Unknown
~7HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1      @9      @A      @I      @at???\8?i?C{*?????Unknown
[8HostAddV2"Adam/add(1      @9      @A      @I      @at???\8?i??v?????Unknown
v9HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @at???\8?i????????Unknown
\:HostArgMax"ArgMax_1(1      @9      @A      @I      @a?nDJH3?i?3???????Unknown
?;HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9       @A      @I       @a?nDJH3?i??A?h????Unknown
?<HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?nDJH3?iEO???????Unknown
y=HostGatherV2"#sequential/dense/Tensordot/GatherV2(1      @9      @A      @I      @a?nDJH3?i???:????Unknown
?>HostReadVariableOp")sequential/dense/Tensordot/ReadVariableOp(1      @9      @A      @I      @a?nDJH3?i?j??????Unknown
y?HostConcatV2"#sequential/dense/Tensordot/concat_1(1      @9      @A      @I      @a?nDJH3?i??c?????Unknown
v@HostReadVariableOp"Adam/Cast_3/ReadVariableOp(1      @9      @A      @I      @aX?fo?,?i?bZ??????Unknown
bAHostDivNoNan"div_no_nan_1(1      @9      @A      @I      @aX?fo?,?i*?P??????Unknown
?BHostTile";gradient_tape/categorical_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @aX?fo?,?i{7GNy????Unknown
?CHostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1      @9      @A      @I      @aX?fo?,?i̡=H????Unknown
rDHostPack" sequential/dense/Tensordot/stack(1      @9      @A      @I      @aX?fo?,?i4?????Unknown
vEHostReadVariableOp"Adam/Cast_2/ReadVariableOp(1       @9       @A       @I       @a?nDJH#?i?R?`K????Unknown
YFHostPow"Adam/Pow(1       @9       @A       @I       @a?nDJH#?iߙ|?????Unknown
[GHostPow"
Adam/Pow_1(1       @9       @A       @I       @a?nDJH#?i?? j?????Unknown
oHHostReadVariableOp"Adam/ReadVariableOp(1       @9       @A       @I       @a?nDJH#?i?'???????Unknown
vIHostAssignAddVariableOp"AssignAddVariableOp_4(1       @9       @A       @I       @a?nDJH#?i?nis????Unknown
VJHostCast"Cast(1       @9       @A       @I       @a?nDJH#?ic??Q????Unknown
XKHostEqual"Equal(1       @9       @A       @I       @a?nDJH#?iD??|?????Unknown
?LHostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1     ?Q@9     ?Q@A       @I       @a?nDJH#?i%CV?????Unknown
`MHostDivNoNan"
div_no_nan(1       @9       @A       @I       @a?nDJH#?i????????Unknown
uNHostReadVariableOp"div_no_nan/ReadVariableOp(1       @9       @A       @I       @a?nDJH#?i?О
$????Unknown
wOHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a?nDJH#?i?C?X????Unknown
?PHostReadVariableOp"'sequential/conv0/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a?nDJH#?i?^??????Unknown
?QHostReadVariableOp"&sequential/conv0/Conv2D/ReadVariableOp(1       @9       @A       @I       @a?nDJH#?i?????????Unknown
?RHostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a?nDJH#?ik?/?????Unknown
?SHostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1       @9       @A       @I       @a?nDJH#?iL3ԡ*????Unknown
]THostCast"Adam/Cast_1(1      ??9      ??A      ??I      ??a?nDJH?i?V&??????Unknown
tUHostAssignAddVariableOp"AssignAddVariableOp(1      ??9      ??A      ??I      ??a?nDJH?i,zx&_????Unknown
vVHostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??a?nDJH?i???h?????Unknown
vWHostAssignAddVariableOp"AssignAddVariableOp_3(1      ??9      ??A      ??I      ??a?nDJH?i???????Unknown
XXHostCast"Cast_1(1      ??9      ??A      ??I      ??a?nDJH?i|?n?-????Unknown
XYHostCast"Cast_2(1      ??9      ??A      ??I      ??a?nDJH?i??/?????Unknown
aZHostIdentity"Identity(1      ??9      ??A      ??I      ??a?nDJH?i\+rb????Unknown?
T[HostMul"Mul(1      ??9      ??A      ??I      ??a?nDJH?i?Ne??????Unknown
{\HostSum"*categorical_crossentropy/weighted_loss/Sum(1      ??9      ??A      ??I      ??a?nDJH?i<r???????Unknown
?]HostCast"8categorical_crossentropy/weighted_loss/num_elements/Cast(1      ??9      ??A      ??I      ??a?nDJH?i??	91????Unknown
?^HostDivNoNan",categorical_crossentropy/weighted_loss/value(1      ??9      ??A      ??I      ??a?nDJH?i?[{?????Unknown
w_HostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a?nDJH?i?ܭ?e????Unknown
s`HostProd"!sequential/dense/Tensordot/Prod_1(1      ??9      ??A      ??I      ??a?nDJH?i?????????Unknown
LaHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(i?????????Unknown
nbHostDivNoNan"Egradient_tape/categorical_crossentropy/weighted_loss/value/div_no_nan(i?????????Unknown2CPU