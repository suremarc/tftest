"?S
BHostIDLE"IDLE1    ???@A    ???@a̜??????i̜???????Unknown
?HostConv2DBackpropFilter":gradient_tape/sequential/conv0/Conv2D/Conv2DBackpropFilter(1     ??@9     ??@A     ??@I     ??@a?i???i.7`?o????Unknown
oHost_FusedConv2D"sequential/conv0/Relu(1     L?@9     L?@A     L?@I     L?@a????????i?U??????Unknown
mHostBiasAdd"sequential/dense/BiasAdd(1     0?@9     0?@A     0?@I     0?@a	|??偢?i???-"$???Unknown
?HostMaxPoolGrad"2gradient_tape/sequential/pool0/MaxPool/MaxPoolGrad(1     ??@9     ??@A     ??@I     ??@aX{u?๕?i?9?5?????Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1     @?@9     @?@A     @?@I     @?@a?uRX????i2?]?Wq???Unknown
?HostMatMul"8gradient_tape/sequential/dense/Tensordot/MatMul/MatMul_1(1     0|@9     0|@A     0|@I     0|@a}????G??iv???????Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1      {@9      {@A      {@I      {@a???=???i?~?ڡ????Unknown
^	HostGatherV2"GatherV2(1     ?x@9     ?x@A     ?x@I     ?x@a[???l??i?8T????Unknown
?
HostResourceApplyAdam"$Adam/Adam/update_4/ResourceApplyAdam(1     ?x@9     ?x@A     ?x@I     ?x@a?N=?
??iW??7~r???Unknown
uHostMatMul"!sequential/dense/Tensordot/MatMul(1     ?x@9     ?x@A     ?x@I     ?x@a?N=?
??i???6?????Unknown
?HostMatMul"6gradient_tape/sequential/dense/Tensordot/MatMul/MatMul(1     ?v@9     ?v@A     ?v@I     ?v@a?s??????icEf?Z???Unknown
mHostMaxPool"sequential/pool0/MaxPool(1     @s@9     @s@A     @s@I     @s@aGPy򈚇?i????????Unknown
?HostBiasAddGrad"2gradient_tape/sequential/conv0/BiasAdd/BiasAddGrad(1     ?h@9     ?h@A     ?h@I     ?h@aȋ?7?X~?i?r~|?????Unknown
tHost_FusedMatMul"sequential/dense_1/BiasAdd(1     ?g@9     ?g@A     ?g@I     ?g@aY[0???|?is?B?H/???Unknown
?HostRandomUniform"7sequential/dropout/dropout/random_uniform/RandomUniform(1     ?c@9     ?c@A     ?c@I     ?c@a)??i?w?i???_???Unknown
}HostReluGrad"'gradient_tape/sequential/conv0/ReluGrad(1     `b@9     `b@A     `b@I     `b@a,?-?߇v?i?(?t*????Unknown
?HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1     @\@9     @\@A     @\@I     @\@a?v??Qq?icxή???Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1     ?X@9     ?X@A     ?X@I     ?X@a?N=?
n?i?S8??????Unknown
}HostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1     ?X@9     ?X@A     ?X@I     ?X@a?N=?
n?i???????Unknown
gHostRelu"sequential/dense/Relu(1     ?W@9     ?W@A     ?W@I     ?W@aY[0???l?i\?ګ????Unknown
?HostGreaterEqual"'sequential/dropout/dropout/GreaterEqual(1     ?S@9     ?S@A     ?S@I     ?S@a??{7h?i&A?'????Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1     ?R@9     ?R@A     ?R@I     ?R@a??r??f?i?????6???Unknown
qHostMul" sequential/dropout/dropout/Mul_1(1      O@9      O@A      O@I      O@a????jc?i?|?(?I???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1     ?I@9     ?I@A     ?I@I     ?I@aqBJ?eD_?iݡ?[?Y???Unknown
?HostSoftmaxCrossEntropyWithLogits":categorical_crossentropy/softmax_cross_entropy_with_logits(1      I@9      I@A      I@I      I@a??C?r?^?i?C??h???Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      H@9      H@A      H@I      H@a?6ьm]?i,?@ۖw???Unknown
gHostStridedSlice"strided_slice(1      H@9      H@A      H@I      H@a?6ьm]?i?z??M????Unknown
`HostGatherV2"
GatherV2_1(1     ?E@9     ?E@A     ?E@I     ?E@aAt&?\Z?iх?|????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1     ?B@9     ?B@A     ?B@I     ?B@a?????V?i?}?Ӟ???Unknown?
?HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1      @@9      @@A      @@I      @@a?8??]?S?i:??Ţ????Unknown
o HostMul"sequential/dropout/dropout/Mul(1      8@9      8@A      8@I      8@a?6ьmM?i?2')?????Unknown
?!HostMul"Lgradient_tape/categorical_crossentropy/softmax_cross_entropy_with_logits/mul(1      7@9      7@A      7@I      7@a??)??3L?ig??????Unknown
\"HostArgMax"ArgMax_1(1      5@9      5@A      5@I      5@a{?7ۿI?if??	{????Unknown
?#HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1      2@9      2@A      2@I      2@a???)F?i????????Unknown
d$HostDataset"Iterator::Model(1      V@9      V@A      *@I      *@a7?P?X???iƥ*??????Unknown
i%HostWriteSummary"WriteSummary(1      *@9      *@A      *@I      *@a7?P?X???i?O<??????Unknown?
?&HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1      (@9      (@A      (@I      (@a?6ьm=?i?v֛?????Unknown
?'HostResourceApplyAdam"$Adam/Adam/update_5/ResourceApplyAdam(1      (@9      (@A      (@I      (@a?6ьm=?i??pMS????Unknown
o(HostSoftmax"sequential/dense_1/Softmax(1      (@9      (@A      (@I      (@a?6ьm=?io?
? ????Unknown
l)HostIteratorGetNext"IteratorGetNext(1      $@9      $@A      $@I      $@a?Y??8?i?䵽????Unknown
q*HostCast"sequential/dropout/dropout/Cast(1      $@9      $@A      $@I      $@a?Y??8?i1a|"????Unknown
?+HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      "@9      "@A      "@I      "@a???)6?iU????????Unknown
[,HostAddV2"Adam/add(1       @9       @A       @I       @a?8??]?3?i<?P?X????Unknown
Z-HostArgMax"ArgMax(1      @9      @A      @I      @a?6ьm-?i??f/????Unknown
e.Host
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @a?6ьm-?i??>????Unknown?
?/HostReadVariableOp")sequential/dense/Tensordot/ReadVariableOp(1      @9      @A      @I      @a?Y??(?iFs@??????Unknown
Y0HostPow"Adam/Pow(1      @9      @A      @I      @a?8??]?#?i:???????Unknown
v1HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a?8??]?#?i.??i????Unknown
{2HostSum"*categorical_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a?8??]?#?i"??O<????Unknown
?3HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?8??]?#?i??5v????Unknown
?4HostReadVariableOp"&sequential/conv0/Conv2D/ReadVariableOp(1      @9      @A      @I      @a?8??]?#?i
???????Unknown
?5HostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a?8??]?#?i??t?????Unknown
q6HostProd"sequential/dense/Tensordot/Prod(1      @9      @A      @I      @a?8??]?#?i??R?#????Unknown
y7HostConcatV2"#sequential/dense/Tensordot/concat_1(1      @9      @A      @I      @a?8??]?#?i??0?]????Unknown
~8HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1      @9      @A      @I      @a?6ьm?i?d?9I????Unknown
]9HostCast"Adam/Cast_1(1      @9      @A      @I      @a?6ьm?iT???4????Unknown
v:HostReadVariableOp"Adam/Cast_2/ReadVariableOp(1      @9      @A      @I      @a?6ьm?ixd ????Unknown
o;HostReadVariableOp"Adam/ReadVariableOp(1      @9      @A      @I      @a?6ьm?i??~????Unknown
t<HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a?6ьm?iy?1??????Unknown
V=HostCast"Cast(1      @9      @A      @I      @a?6ьm?i0?W?????Unknown
V>HostSum"Sum_2(1      @9      @A      @I      @a?6ьm?i?????????Unknown
??HostTile";gradient_tape/categorical_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a?6ьm?i?(e0?????Unknown
y@HostGatherV2"#sequential/dense/Tensordot/GatherV2(1      @9      @A      @I      @a?6ьm?iU?˜?????Unknown
tAHostReadVariableOp"Adam/Cast/ReadVariableOp(1       @9       @A       @I       @a?8??]??iϸ??A????Unknown
[BHostPow"
Adam/Pow_1(1       @9       @A       @I       @a?8??]??iI????????Unknown
?CHostReadVariableOp"'sequential/conv0/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a?8??]??i?Řu{????Unknown
rDHostPack" sequential/dense/Tensordot/stack(1       @9       @A       @I       @a?8??]??i=̇h????Unknown
?EHostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a?8??]??i??v[?????Unknown
?FHostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1       @9       @A       @I       @a?8??]??i1?eNR????Unknown
vGHostReadVariableOp"Adam/Cast_3/ReadVariableOp(1      ??9      ??A      ??I      ??a?8??]??in\?Ǡ????Unknown
vHHostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??a?8??]??i??TA?????Unknown
vIHostAssignAddVariableOp"AssignAddVariableOp_3(1      ??9      ??A      ??I      ??a?8??]??i?b̺=????Unknown
vJHostAssignAddVariableOp"AssignAddVariableOp_4(1      ??9      ??A      ??I      ??a?8??]??i%?C4?????Unknown
XKHostCast"Cast_1(1      ??9      ??A      ??I      ??a?8??]??ibi???????Unknown
XLHostEqual"Equal(1      ??9      ??A      ??I      ??a?8??]??i??2')????Unknown
TMHostMul"Mul(1      ??9      ??A      ??I      ??a?8??]??i?o??w????Unknown
?NHostDivNoNan",categorical_crossentropy/weighted_loss/value(1      ??9      ??A      ??I      ??a?8??]??i?!?????Unknown
`OHostDivNoNan"
div_no_nan(1      ??9      ??A      ??I      ??a?8??]??iVv??????Unknown
uPHostReadVariableOp"div_no_nan/ReadVariableOp(1      ??9      ??A      ??I      ??a?8??]??i??c????Unknown
bQHostDivNoNan"div_no_nan_1(1      ??9      ??A      ??I      ??a?8??]??i?|???????Unknown
?RHostDivNoNan"Egradient_tape/categorical_crossentropy/weighted_loss/value/div_no_nan(1      ??9      ??A      ??I      ??a?8??]??i     ???Unknown
{SHostGatherV2"%sequential/dense/Tensordot/GatherV2_1(1      ??9      ??A      ??I      ??a?8??]??i???<' ???Unknown
sTHostProd"!sequential/dense/Tensordot/Prod_1(1      ??9      ??A      ??I      ??a?8??]??iB?wyN ???Unknown
+UHostCast"Cast_2(iB?wyN ???Unknown
4VHostIdentity"Identity(iB?wyN ???Unknown?
iWHostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(iB?wyN ???Unknown
]XHostCast"8categorical_crossentropy/weighted_loss/num_elements/Cast(iB?wyN ???Unknown
JYHostReadVariableOp"div_no_nan/ReadVariableOp_1(iB?wyN ???Unknown
JZHostReadVariableOp"div_no_nan_1/ReadVariableOp(iB?wyN ???Unknown
L[HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(iB?wyN ???Unknown*?R
?HostConv2DBackpropFilter":gradient_tape/sequential/conv0/Conv2D/Conv2DBackpropFilter(1     ??@9     ??@A     ??@I     ??@a?!Uu???i?!Uu????Unknown
oHost_FusedConv2D"sequential/conv0/Relu(1     L?@9     L?@A     L?@I     L?@aر8ԑR??i_N)?R???Unknown
mHostBiasAdd"sequential/dense/BiasAdd(1     0?@9     0?@A     0?@I     0?@a`?? і??i|?-?????Unknown
?HostMaxPoolGrad"2gradient_tape/sequential/pool0/MaxPool/MaxPoolGrad(1     ??@9     ??@A     ??@I     ??@aD6??z+??i??س?>???Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1     @?@9     @?@A     @?@I     @?@a/?/??*??iÂ?H????Unknown
?HostMatMul"8gradient_tape/sequential/dense/Tensordot/MatMul/MatMul_1(1     0|@9     0|@A     0|@I     0|@a?T?v?9??ihJ??????Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1      {@9      {@A      {@I      {@ap??????i%?%`?????Unknown
^HostGatherV2"GatherV2(1     ?x@9     ?x@A     ?x@I     ?x@a??K?S???i@
?????Unknown
?	HostResourceApplyAdam"$Adam/Adam/update_4/ResourceApplyAdam(1     ?x@9     ?x@A     ?x@I     ?x@a~???õ??i?X?????Unknown
u
HostMatMul"!sequential/dense/Tensordot/MatMul(1     ?x@9     ?x@A     ?x@I     ?x@a~???õ??ioq"a"???Unknown
?HostMatMul"6gradient_tape/sequential/dense/Tensordot/MatMul/MatMul(1     ?v@9     ?v@A     ?v@I     ?v@a????>??i?uiS???Unknown
mHostMaxPool"sequential/pool0/MaxPool(1     @s@9     @s@A     @s@I     @s@a4g?]3B??i(?W?d????Unknown
?HostBiasAddGrad"2gradient_tape/sequential/conv0/BiasAdd/BiasAddGrad(1     ?h@9     ?h@A     ?h@I     ?h@a3??)j???i?p??ot???Unknown
tHost_FusedMatMul"sequential/dense_1/BiasAdd(1     ?g@9     ?g@A     ?g@I     ?g@a???S)??i?F;?????Unknown
?HostRandomUniform"7sequential/dropout/dropout/random_uniform/RandomUniform(1     ?c@9     ?c@A     ?c@I     ?c@a?J9?????i!?F=_???Unknown
}HostReluGrad"'gradient_tape/sequential/conv0/ReluGrad(1     `b@9     `b@A     `b@I     `b@a=K"e???iN}??Q????Unknown
?HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1     @\@9     @\@A     @\@I     @\@a!1=?D??irH?c???Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1     ?X@9     ?X@A     ?X@I     ?X@a~???õ??i1x??:S???Unknown
}HostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1     ?X@9     ?X@A     ?X@I     ?X@a~???õ??iO~????Unknown
gHostRelu"sequential/dense/Relu(1     ?W@9     ?W@A     ?W@I     ?W@a???S)??iR?]?.????Unknown
?HostGreaterEqual"'sequential/dropout/dropout/GreaterEqual(1     ?S@9     ?S@A     ?S@I     ?S@a
.????z?i?-?H???Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1     ?R@9     ?R@A     ?R@I     ?R@a]????y?i???z7????Unknown
qHostMul" sequential/dropout/dropout/Mul_1(1      O@9      O@A      O@I      O@a??X?$u?im?׀i???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1     ?I@9     ?I@A     ?I@I     ?I@aTN??]dq?i??A?I????Unknown
?HostSoftmaxCrossEntropyWithLogits":categorical_crossentropy/softmax_cross_entropy_with_logits(1      I@9      I@A      I@I      I@a?j8?q?i?<??c????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      H@9      H@A      H@I      H@a?Jvv^p?i?Ѥ? ????Unknown
gHostStridedSlice"strided_slice(1      H@9      H@A      H@I      H@a?Jvv^p?ig???????Unknown
`HostGatherV2"
GatherV2_1(1     ?E@9     ?E@A     ?E@I     ?E@a?e???Sm?i?W%x1???Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1     ?B@9     ?B@A     ?B@I     ?B@a??]?K<i?i=??m&???Unknown?
?HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1      @@9      @@A      @@I      @@a?ڸ?H?e?in?A<???Unknown
oHostMul"sequential/dropout/dropout/Mul(1      8@9      8@A      8@I      8@a?Jvv^`?i??/??L???Unknown
? HostMul"Lgradient_tape/categorical_crossentropy/softmax_cross_entropy_with_logits/mul(1      7@9      7@A      7@I      7@ay??b?__?i?a_O\???Unknown
\!HostArgMax"ArgMax_1(1      5@9      5@A      5@I      5@a!?OO?\?i???j???Unknown
?"HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1      2@9      2@A      2@I      2@a?o???X?i?N???v???Unknown
d#HostDataset"Iterator::Model(1      V@9      V@A      *@I      *@a?1& ??Q?i?aa?????Unknown
i$HostWriteSummary"WriteSummary(1      *@9      *@A      *@I      *@a?1& ??Q?iuኤ????Unknown?
?%HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1      (@9      (@A      (@I      (@a?Jvv^P?ih??Ӑ???Unknown
?&HostResourceApplyAdam"$Adam/Adam/update_5/ResourceApplyAdam(1      (@9      (@A      (@I      (@a?Jvv^P?i??W????Unknown
o'HostSoftmax"sequential/dense_1/Softmax(1      (@9      (@A      (@I      (@a?Jvv^P?i??<2????Unknown
l(HostIteratorGetNext"IteratorGetNext(1      $@9      $@A      $@I      $@au'?HK?i?.DC????Unknown
q)HostCast"sequential/dropout/dropout/Cast(1      $@9      $@A      $@I      $@au'?HK?i?x?I֮???Unknown
?*HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      "@9      "@A      "@I      "@a?o???H?i??a??????Unknown
[+HostAddV2"Adam/add(1       @9       @A       @I       @a?ڸ?H?E?i?B??n????Unknown
Z,HostArgMax"ArgMax(1      @9      @A      @I      @a?Jvv^@?ir?&&?????Unknown
e-Host
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @a?Jvv^@?ih?Ý????Unknown?
?.HostReadVariableOp")sequential/dense/Tensordot/ReadVariableOp(1      @9      @A      @I      @au'?H;?i??????Unknown
Y/HostPow"Adam/Pow(1      @9      @A      @I      @a?ڸ?H?5?i?00?????Unknown
v0HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a?ڸ?H?5?i3{D?{????Unknown
{1HostSum"*categorical_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a?ڸ?H?5?iN2X6????Unknown
?2HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?ڸ?H?5?ii?kk?????Unknown
?3HostReadVariableOp"&sequential/conv0/Conv2D/ReadVariableOp(1      @9      @A      @I      @a?ڸ?H?5?i??Ԫ????Unknown
?4HostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a?ڸ?H?5?i?W?=e????Unknown
q5HostProd"sequential/dense/Tensordot/Prod(1      @9      @A      @I      @a?ڸ?H?5?i???????Unknown
y6HostConcatV2"#sequential/dense/Tensordot/concat_1(1      @9      @A      @I      @a?ڸ?H?5?i?ź?????Unknown
~7HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1      @9      @A      @I      @a?Jvv^0?i*????????Unknown
]8HostCast"Adam/Cast_1(1      @9      @A      @I      @a?Jvv^0?iXX??????Unknown
v9HostReadVariableOp"Adam/Cast_2/ReadVariableOp(1      @9      @A      @I      @a?Jvv^0?i?!'|?????Unknown
o:HostReadVariableOp"Adam/ReadVariableOp(1      @9      @A      @I      @a?Jvv^0?i)??J	????Unknown
t;HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a?Jvv^0?i~??????Unknown
V<HostCast"Cast(1      @9      @A      @I      @a?Jvv^0?i?}?? ????Unknown
V=HostSum"Sum_2(1      @9      @A      @I      @a?Jvv^0?i(Gb?,????Unknown
?>HostTile";gradient_tape/categorical_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a?Jvv^0?i}1?8????Unknown
y?HostGatherV2"#sequential/dense/Tensordot/GatherV2(1      @9      @A      @I      @a?Jvv^0?i???TD????Unknown
t@HostReadVariableOp"Adam/Cast/ReadVariableOp(1       @9       @A       @I       @a?ڸ?H?%?i`????????Unknown
[AHostPow"
Adam/Pow_1(1       @9       @A       @I       @a?ڸ?H?%?i????????Unknown
?BHostReadVariableOp"'sequential/conv0/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a?ڸ?H?%?i|l??[????Unknown
rCHostPack" sequential/dense/Tensordot/stack(1       @9       @A       @I       @a?ڸ?H?%?i
H''?????Unknown
?DHostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a?ڸ?H?%?i?#?[????Unknown
?EHostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1       @9       @A       @I       @a?ڸ?H?%?i&?:?s????Unknown
vFHostReadVariableOp"Adam/Cast_3/ReadVariableOp(1      ??9      ??A      ??I      ??a?ڸ?H??i??*"????Unknown
vGHostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??a?ڸ?H??i?????????Unknown
vHHostAssignAddVariableOp"AssignAddVariableOp_3(1      ??9      ??A      ??I      ??a?ڸ?H??i{?	_????Unknown
vIHostAssignAddVariableOp"AssignAddVariableOp_4(1      ??9      ??A      ??I      ??a?ڸ?H??iB?N?-????Unknown
XJHostCast"Cast_1(1      ??9      ??A      ??I      ??a?ڸ?H??i	????????Unknown
XKHostEqual"Equal(1      ??9      ??A      ??I      ??a?ڸ?H??iБ?-?????Unknown
TLHostMul"Mul(1      ??9      ??A      ??I      ??a?ڸ?H??i??9????Unknown
?MHostDivNoNan",categorical_crossentropy/weighted_loss/value(1      ??9      ??A      ??I      ??a?ڸ?H??i^mbb?????Unknown
`NHostDivNoNan"
div_no_nan(1      ??9      ??A      ??I      ??a?ڸ?H??i%[???????Unknown
uOHostReadVariableOp"div_no_nan/ReadVariableOp(1      ??9      ??A      ??I      ??a?ڸ?H??i?H??E????Unknown
bPHostDivNoNan"div_no_nan_1(1      ??9      ??A      ??I      ??a?ڸ?H??i?611?????Unknown
?QHostDivNoNan"Egradient_tape/categorical_crossentropy/weighted_loss/value/div_no_nan(1      ??9      ??A      ??I      ??a?ڸ?H??iz$vˢ????Unknown
{RHostGatherV2"%sequential/dense/Tensordot/GatherV2_1(1      ??9      ??A      ??I      ??a?ڸ?H??iA?eQ????Unknown
sSHostProd"!sequential/dense/Tensordot/Prod_1(1      ??9      ??A      ??I      ??a?ڸ?H??i     ???Unknown
+THostCast"Cast_2(i     ???Unknown
4UHostIdentity"Identity(i     ???Unknown?
iVHostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(i     ???Unknown
]WHostCast"8categorical_crossentropy/weighted_loss/num_elements/Cast(i     ???Unknown
JXHostReadVariableOp"div_no_nan/ReadVariableOp_1(i     ???Unknown
JYHostReadVariableOp"div_no_nan_1/ReadVariableOp(i     ???Unknown
LZHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(i     ???Unknown2CPU