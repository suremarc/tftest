"?U
BHostIDLE"IDLE1    ??@A    ??@a?j???+??i?j???+???Unknown
?HostConv2DBackpropFilter":gradient_tape/sequential/conv0/Conv2D/Conv2DBackpropFilter(1     ?@9     ?@A     ?@I     ?@a???m???ib?g?˓???Unknown
?HostConv2DBackpropInput"9gradient_tape/sequential/conv1/Conv2D/Conv2DBackpropInput(1     Ў@9     Ў@A     Ў@I     Ў@az$XAj??i?G}0?????Unknown
?HostConv2DBackpropFilter":gradient_tape/sequential/conv1/Conv2D/Conv2DBackpropFilter(1     ?@9     ?@A     ?@I     ?@aTЛ98??i?L:?e????Unknown
oHost_FusedConv2D"sequential/conv0/Relu(1     `?@9     `?@A     `?@I     `?@a?˄???i????V????Unknown
?HostMaxPoolGrad"2gradient_tape/sequential/pool2/MaxPool/MaxPoolGrad(1     ?|@9     ?|@A     ?|@I     ?|@av??ν??i???fE????Unknown
?HostMaxPoolGrad"2gradient_tape/sequential/pool0/MaxPool/MaxPoolGrad(1     ?y@9     ?y@A     ?y@I     ?y@a?Z?˘??i???,????Unknown
?HostBiasAddGrad"2gradient_tape/sequential/conv0/BiasAdd/BiasAddGrad(1     ?y@9     ?y@A     ?y@I     ?y@a?x
ə?i~???Z????Unknown
m	HostMaxPool"sequential/pool0/MaxPool(1     pv@9     pv@A     pv@I     pv@a?tb?????i%???8???Unknown
?
HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      m@9      m@A     ?l@I     ?l@a6O?{???ibG?=?????Unknown
oHost_FusedConv2D"sequential/conv1/Relu(1     ?f@9     ?f@A     ?f@I     ?f@a*?G????i?e??C???Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      e@9      e@A      e@I      e@a(dˬ?!??i|?fJ?[???Unknown
mHostMaxPool"sequential/pool2/MaxPool(1     ?d@9     ?d@A     ?d@I     ?d@a'??eࠄ?ie??M????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_4/ResourceApplyAdam(1     @^@9     @^@A     @^@I     @^@a9?IQ?p~?i???.????Unknown
}HostReluGrad"'gradient_tape/sequential/conv0/ReluGrad(1     @^@9     @^@A     @^@I     @^@a9?IQ?p~?i?AB?(???Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1      ]@9      ]@A      ]@I      ]@a7b?.}?i???lb???Unknown
?HostRandomUniform"7sequential/dropout/dropout/random_uniform/RandomUniform(1     ?Q@9     ?Q@A     ?Q@I     ?Q@a!~???q?i?X???????Unknown
^HostGatherV2"GatherV2(1      P@9      P@A      P@I      P@a@-??p?i??Pإ???Unknown
?HostSoftmaxCrossEntropyWithLogits":categorical_crossentropy/softmax_cross_entropy_with_logits(1     ?M@9     ?M@A     ?M@I     ?M@a8n?fO?m?i?>'??????Unknown
?HostGreaterEqual"'sequential/dropout/dropout/GreaterEqual(1      M@9      M@A      M@I      M@a7b?.m?i??F"?????Unknown
?HostRandomUniform"9sequential/dropout_1/dropout/random_uniform/RandomUniform(1      F@9      F@A      F@I      F@a*8;H#f?iǾ?j?????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1      E@9      E@A      E@I      E@a(dˬ?!e?i+?.????Unknown?
dHostDataset"Iterator::Model(1     ?b@9     ?b@A     ?@@I     ?@@a?V,v?`?i??Z?????Unknown
`HostDivNoNan"
div_no_nan(1     ?@@9     ?@@A     ?@@I     ?@@a?V,v?`?i7?0-???Unknown
?HostBiasAddGrad"2gradient_tape/sequential/conv1/BiasAdd/BiasAddGrad(1     ?@@9     ?@@A     ?@@I     ?@@a?V,v?`?i)??z?=???Unknown
?HostReadVariableOp"'sequential/conv0/BiasAdd/ReadVariableOp(1     ?@@9     ?@@A     ?@@I     ?@@a?V,v?`?i????dN???Unknown
qHostMul" sequential/dropout/dropout/Mul_1(1     ?@@9     ?@@A     ?@@I     ?@@a?V,v?`?i};g?^???Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1      5@9      5@A      5@I      5@a(dˬ?!U?i/??=?i???Unknown
{HostMatMul"'gradient_tape/sequential/dense/MatMul_1(1      5@9      5@A      5@I      5@a(dˬ?!U?i??!t???Unknown
?HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1      2@9      2@A      2@I      2@a"???R?iU???/}???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1      1@9      1@A      1@I      1@a ?sCQ?i_?s%?????Unknown
r Host_FusedMatMul"sequential/dense/BiasAdd(1      0@9      0@A      0@I      0@a@-??P?i?F??ɍ???Unknown
i!HostWriteSummary"WriteSummary(1      ,@9      ,@A      ,@I      ,@a50??,L?iˊ?3Ք???Unknown?
?"HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1      *@9      *@A      *@I      *@a1?it?)J?i-?g?_????Unknown
?#HostResourceApplyAdam"$Adam/Adam/update_5/ResourceApplyAdam(1      *@9      *@A      *@I      *@a1?it?)J?i???????Unknown
m$HostSoftmax"sequential/dense/Softmax(1      &@9      &@A      &@I      &@a*8;H#F?i??r????Unknown
?%HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1      $@9      $@A      $@I      $@a&?x D?iA%??z????Unknown
}&HostReluGrad"'gradient_tape/sequential/conv1/ReluGrad(1      $@9      $@A      $@I      $@a&?x D?ieâ肱???Unknown
q'HostCast"sequential/dropout/dropout/Cast(1      $@9      $@A      $@I      $@a&?x D?i?aj튶???Unknown
`(HostGatherV2"
GatherV2_1(1      "@9      "@A      "@I      "@a"???B?iC??$????Unknown
l)HostIteratorGetNext"IteratorGetNext(1      "@9      "@A      "@I      "@a"???B?i?Jk\?????Unknown
?*HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @a50??,<?i?l]?????Unknown
?+HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a50??,<?iɎO??????Unknown
g,HostStridedSlice"strided_slice(1      @9      @A      @I      @a50??,<?i??A3*????Unknown
Z-HostArgMax"ArgMax(1      @9      @A      @I      @a-??W}&8?i+??/????Unknown
?.HostMul"Lgradient_tape/categorical_crossentropy/softmax_cross_entropy_with_logits/mul(1      @9      @A      @I      @a-??W}&8?i????3????Unknown
e/Host
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @a&?x 4?i?p?Է????Unknown?
o0HostMul"sequential/dropout/dropout/Mul(1      @9      @A      @I      @a&?x 4?i??_?;????Unknown
[1HostAddV2"Adam/add(1      @9      @A      @I      @a@-??0?is?{?????Unknown
\2HostArgMax"ArgMax_1(1      @9      @A      @I      @a@-??0?i??AB????Unknown
v3HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a@-??0?i?0?vE????Unknown
?4HostTile";gradient_tape/categorical_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a@-??0?ik?ѫH????Unknown
q5HostMul" sequential/dropout_1/dropout/Mul(1      @9      @A      @I      @a@-??0?i|??K????Unknown
v6HostReadVariableOp"Adam/Cast_2/ReadVariableOp(1      @9      @A      @I      @a-??W}&(?iQ??H?????Unknown
v7HostReadVariableOp"Adam/Cast_3/ReadVariableOp(1      @9      @A      @I      @a-??W}&(?i?t??P????Unknown
o8HostReadVariableOp"Adam/ReadVariableOp(1      @9      @A      @I      @a-??W}&(?i??n?????Unknown
V9HostCast"Cast(1      @9      @A      @I      @a-??W}&(?imD?U????Unknown
V:HostSum"Sum_2(1      @9      @A      @I      @a-??W}&(?iI???????Unknown
{;HostSum"*categorical_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a-??W}&(?i?e?OZ????Unknown
?<HostReadVariableOp"&sequential/conv0/Conv2D/ReadVariableOp(1      @9      @A      @I      @a-??W}&(?i??ķ?????Unknown
?=HostReadVariableOp"'sequential/conv1/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a-??W}&(?i^?_????Unknown
s>HostCast"!sequential/dropout_1/dropout/Cast(1      @9      @A      @I      @a-??W}&(?iA?o??????Unknown
??HostGreaterEqual")sequential/dropout_1/dropout/GreaterEqual(1      @9      @A      @I      @a-??W}&(?iVE?c????Unknown
~@HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1       @9       @A       @I       @a@-?? ?iS?Ӊe????Unknown
tAHostReadVariableOp"Adam/Cast/ReadVariableOp(1       @9       @A       @I       @a@-?? ?i'?a$g????Unknown
[BHostPow"
Adam/Pow_1(1       @9       @A       @I       @a@-?? ?i?N??h????Unknown
tCHostAssignAddVariableOp"AssignAddVariableOp(1       @9       @A       @I       @a@-?? ?iϡ~Yj????Unknown
XDHostEqual"Equal(1       @9       @A       @I       @a@-?? ?i???k????Unknown
?EHostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1       @9       @A       @I       @a@-?? ?iwG??m????Unknown
?FHostMul"0gradient_tape/sequential/dropout_1/dropout/Mul_1(1       @9       @A       @I       @a@-?? ?iK?))o????Unknown
?GHostReadVariableOp"&sequential/conv1/Conv2D/ReadVariableOp(1       @9       @A       @I       @a@-?? ?i???p????Unknown
?HHostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a@-?? ?i??F^r????Unknown
?IHostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1       @9       @A       @I       @a@-?? ?iǒ??s????Unknown
sJHostMul""sequential/dropout_1/dropout/Mul_1(1       @9       @A       @I       @a@-?? ?i??b?u????Unknown
]KHostCast"Adam/Cast_1(1      ??9      ??A      ??I      ??a@-???i?`?????Unknown
vLHostAssignAddVariableOp"AssignAddVariableOp_4(1      ??9      ??A      ??I      ??a@-???io8?-w????Unknown
XMHostCast"Cast_1(1      ??9      ??A      ??I      ??a@-???i?a8??????Unknown
XNHostCast"Cast_2(1      ??9      ??A      ??I      ??a@-???iC??x????Unknown
aOHostIdentity"Identity(1      ??9      ??A      ??I      ??a@-???i??ƕ?????Unknown?
?PHostCast"8categorical_crossentropy/weighted_loss/num_elements/Cast(1      ??9      ??A      ??I      ??a@-???i?cz????Unknown
?QHostDivNoNan",categorical_crossentropy/weighted_loss/value(1      ??9      ??A      ??I      ??a@-???i?U0?????Unknown
uRHostReadVariableOp"div_no_nan/ReadVariableOp(1      ??9      ??A      ??I      ??a@-???i?0??{????Unknown
wSHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a@-???iUZ???????Unknown
bTHostDivNoNan"div_no_nan_1(1      ??9      ??A      ??I      ??a@-???i??*?}????Unknown
yUHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a@-???i)?qe?????Unknown
?VHostDivNoNan"Egradient_tape/categorical_crossentropy/weighted_loss/value/div_no_nan(1      ??9      ??A      ??I      ??a@-???i?ָ2????Unknown
WHostMul".gradient_tape/sequential/dropout_1/dropout/Mul(1      ??9      ??A      ??I      ??a@-???i?????????Unknown
,XHostPow"Adam/Pow(i?????????Unknown
IYHostAssignAddVariableOp"AssignAddVariableOp_1(i?????????Unknown
IZHostAssignAddVariableOp"AssignAddVariableOp_3(i?????????Unknown
'[HostMul"Mul(i?????????Unknown
J\HostReadVariableOp"div_no_nan_1/ReadVariableOp(i?????????Unknown*?U
?HostConv2DBackpropFilter":gradient_tape/sequential/conv0/Conv2D/Conv2DBackpropFilter(1     ?@9     ?@A     ?@I     ?@a?9of??i?9of???Unknown
?HostConv2DBackpropInput"9gradient_tape/sequential/conv1/Conv2D/Conv2DBackpropInput(1     Ў@9     Ў@A     Ў@I     Ў@a????4???i??j?Q???Unknown
?HostConv2DBackpropFilter":gradient_tape/sequential/conv1/Conv2D/Conv2DBackpropFilter(1     ?@9     ?@A     ?@I     ?@a ???ѵ?i?ϐ??????Unknown
oHost_FusedConv2D"sequential/conv0/Relu(1     `?@9     `?@A     `?@I     `?@a?윧T	??i?<w????Unknown
?HostMaxPoolGrad"2gradient_tape/sequential/pool2/MaxPool/MaxPoolGrad(1     ?|@9     ?|@A     ?|@I     ?|@a?p7??i?|`?????Unknown
?HostMaxPoolGrad"2gradient_tape/sequential/pool0/MaxPool/MaxPoolGrad(1     ?y@9     ?y@A     ?y@I     ?y@a????d)??i.?-?=????Unknown
?HostBiasAddGrad"2gradient_tape/sequential/conv0/BiasAdd/BiasAddGrad(1     ?y@9     ?y@A     ?y@I     ?y@a|A???ҫ?iFL?:fS???Unknown
mHostMaxPool"sequential/pool0/MaxPool(1     pv@9     pv@A     pv@I     pv@a?{t?\??i???2/????Unknown
?	HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      m@9      m@A     ?l@I     ?l@a??*?"7??i2UyH?????Unknown
o
Host_FusedConv2D"sequential/conv1/Relu(1     ?f@9     ?f@A     ?f@I     ?f@a(?!??m??i?a
?W????Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      e@9      e@A      e@I      e@aH??? ͖?i??^ſL???Unknown
mHostMaxPool"sequential/pool2/MaxPool(1     ?d@9     ?d@A     ?d@I     ?d@a??Q?B??i]I??????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_4/ResourceApplyAdam(1     @^@9     @^@A     @^@I     @^@aXacGl??ih9??0????Unknown
}HostReluGrad"'gradient_tape/sequential/conv0/ReluGrad(1     @^@9     @^@A     @^@I     @^@aXacGl??isT??????Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1      ]@9      ]@A      ]@I      ]@a??Ľ?|??ih? ?????Unknown
?HostRandomUniform"7sequential/dropout/dropout/random_uniform/RandomUniform(1     ?Q@9     ?Q@A     ?Q@I     ?Q@a< n+ ??i??n??????Unknown
^HostGatherV2"GatherV2(1      P@9      P@A      P@I      P@a[I~?=_??i??? ???Unknown
?HostSoftmaxCrossEntropyWithLogits":categorical_crossentropy/softmax_cross_entropy_with_logits(1     ?M@9     ?M@A     ?M@I     ?M@a?k|????i\?^?U???Unknown
?HostGreaterEqual"'sequential/dropout/dropout/GreaterEqual(1      M@9      M@A      M@I      M@a??Ľ?|?i&E?	????Unknown
?HostRandomUniform"9sequential/dropout_1/dropout/random_uniform/RandomUniform(1      F@9      F@A      F@I      F@aݤ????w?ip d?????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1      E@9      E@A      E@I      E@aH??? ?v?i?+9i????Unknown?
dHostDataset"Iterator::Model(1     ?b@9     ?b@A     ?@@I     ?@@a?;??7?q?ih?`r=???Unknown
`HostDivNoNan"
div_no_nan(1     ?@@9     ?@@A     ?@@I     ?@@a?;??7?q?i????9???Unknown
?HostBiasAddGrad"2gradient_tape/sequential/conv1/BiasAdd/BiasAddGrad(1     ?@@9     ?@@A     ?@@I     ?@@a?;??7?q?iVY?P?\???Unknown
?HostReadVariableOp"'sequential/conv0/BiasAdd/ReadVariableOp(1     ?@@9     ?@@A     ?@@I     ?@@a?;??7?q?iͽֿ?????Unknown
qHostMul" sequential/dropout/dropout/Mul_1(1     ?@@9     ?@@A     ?@@I     ?@@a?;??7?q?iD"?.?????Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1      5@9      5@A      5@I      5@aH??? ?f?i??/\????Unknown
{HostMatMul"'gradient_tape/sequential/dense/MatMul_1(1      5@9      5@A      5@I      5@aH??? ?f?i?-?0)????Unknown
?HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1      2@9      2@A      2@I      2@a?N[%?c?i?{.V?????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1      1@9      1@A      1@I      1@a?-??1ub?ib??)????Unknown
rHost_FusedMatMul"sequential/dense/BiasAdd(1      0@9      0@A      0@I      0@a[I~?=_a?iN?Uň	???Unknown
i HostWriteSummary"WriteSummary(1      ,@9      ,@A      ,@I      ,@a` ]??f^?iΎG????Unknown?
?!HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1      *@9      *@A      *@I      *@a47?.?:\?ij?^}?&???Unknown
?"HostResourceApplyAdam"$Adam/Adam/update_5/ResourceApplyAdam(1      *@9      *@A      *@I      *@a47?.?:\?iv??4???Unknown
m#HostSoftmax"sequential/dense/Softmax(1      &@9      &@A      &@I      &@aݤ????W?iؒ?Y?@???Unknown
?$HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1      $@9      $@A      $@I      $@a???U?iơ`??K???Unknown
}%HostReluGrad"'gradient_tape/sequential/conv1/ReluGrad(1      $@9      $@A      $@I      $@a???U?i???f?V???Unknown
q&HostCast"sequential/dropout/dropout/Cast(1      $@9      $@A      $@I      $@a???U?i??p?za???Unknown
`'HostGatherV2"
GatherV2_1(1      "@9      "@A      "@I      "@a?N[%?S?i?f?@k???Unknown
l(HostIteratorGetNext"IteratorGetNext(1      "@9      "@A      "@I      "@a?N[%?S?i??u???Unknown
?)HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @a` ]??fN?i??Ľ?|???Unknown
?*HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a` ]??fN?i4??h9????Unknown
g+HostStridedSlice"strided_slice(1      @9      @A      @I      @a` ]??fN?it??Ӌ???Unknown
Z,HostArgMax"ArgMax(1      @9      @A      @I      @a	n?y?J?i???V????Unknown
?-HostMul"Lgradient_tape/categorical_crossentropy/softmax_cross_entropy_with_logits/mul(1      @9      @A      @I      @a	n?y?J?i,r??ژ???Unknown
e.Host
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @a???E?i?y7EH????Unknown?
o/HostMul"sequential/dropout/dropout/Mul(1      @9      @A      @I      @a???E?i?{?????Unknown
[0HostAddV2"Adam/add(1      @9      @A      @I      @a[I~?=_A?i? ??????Unknown
\1HostArgMax"ArgMax_1(1      @9      @A      @I      @a[I~?=_A?i>?N?e????Unknown
v2HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a[I~?=_A?i?_?v?????Unknown
?3HostTile";gradient_tape/categorical_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a[I~?=_A?ib?!F????Unknown
q4HostMul" sequential/dropout_1/dropout/Mul(1      @9      @A      @I      @a[I~?=_A?i???m????Unknown
v5HostReadVariableOp"Adam/Cast_2/ReadVariableOp(1      @9      @A      @I      @a	n?y?:?i????????Unknown
v6HostReadVariableOp"Adam/Cast_3/ReadVariableOp(1      @9      @A      @I      @a	n?y?:?iP???????Unknown
o7HostReadVariableOp"Adam/ReadVariableOp(1      @9      @A      @I      @a	n?y?:?i?E9?2????Unknown
V8HostCast"Cast(1      @9      @A      @I      @a	n?y?:?i?}ȃt????Unknown
V9HostSum"Sum_2(1      @9      @A      @I      @a	n?y?:?iZ?W_?????Unknown
{:HostSum"*categorical_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a	n?y?:?i??:?????Unknown
?;HostReadVariableOp"&sequential/conv0/Conv2D/ReadVariableOp(1      @9      @A      @I      @a	n?y?:?i?$v:????Unknown
?<HostReadVariableOp"'sequential/conv1/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a	n?y?:?id\?{????Unknown
s=HostCast"!sequential/dropout_1/dropout/Cast(1      @9      @A      @I      @a	n?y?:?i??ͽ????Unknown
?>HostGreaterEqual")sequential/dropout_1/dropout/GreaterEqual(1      @9      @A      @I      @a	n?y?:?i??#??????Unknown
~?HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1       @9       @A       @I       @a[I~?=_1?i??ؐ+????Unknown
t@HostReadVariableOp"Adam/Cast/ReadVariableOp(1       @9       @A       @I       @a[I~?=_1?iRk?xW????Unknown
[AHostPow"
Adam/Pow_1(1       @9       @A       @I       @a[I~?=_1?i;B`?????Unknown
tBHostAssignAddVariableOp"AssignAddVariableOp(1       @9       @A       @I       @a[I~?=_1?i?
?G?????Unknown
XCHostEqual"Equal(1       @9       @A       @I       @a[I~?=_1?i?ګ/?????Unknown
?DHostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1       @9       @A       @I       @a[I~?=_1?iv?`????Unknown
?EHostMul"0gradient_tape/sequential/dropout_1/dropout/Mul_1(1       @9       @A       @I       @a[I~?=_1?i?z?2????Unknown
?FHostReadVariableOp"&sequential/conv1/Conv2D/ReadVariableOp(1       @9       @A       @I       @a[I~?=_1?iJ??^????Unknown
?GHostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a[I~?=_1?i?Ί????Unknown
?HHostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1       @9       @A       @I       @a[I~?=_1?i??3??????Unknown
sIHostMul""sequential/dropout_1/dropout/Mul_1(1       @9       @A       @I       @a[I~?=_1?ic????????Unknown
]JHostCast"Adam/Cast_1(1      ??9      ??A      ??I      ??a[I~?=_!?iH!Ñ?????Unknown
vKHostAssignAddVariableOp"AssignAddVariableOp_4(1      ??9      ??A      ??I      ??a[I~?=_!?i-???????Unknown
XLHostCast"Cast_1(1      ??9      ??A      ??I      ??a[I~?=_!?i?wy$????Unknown
XMHostCast"Cast_2(1      ??9      ??A      ??I      ??a[I~?=_!?i?XRm:????Unknown
aNHostIdentity"Identity(1      ??9      ??A      ??I      ??a[I~?=_!?i??,aP????Unknown?
?OHostCast"8categorical_crossentropy/weighted_loss/num_elements/Cast(1      ??9      ??A      ??I      ??a[I~?=_!?i?(Uf????Unknown
?PHostDivNoNan",categorical_crossentropy/weighted_loss/value(1      ??9      ??A      ??I      ??a[I~?=_!?i???H|????Unknown
uQHostReadVariableOp"div_no_nan/ReadVariableOp(1      ??9      ??A      ??I      ??a[I~?=_!?i???<?????Unknown
wRHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a[I~?=_!?ip`?0?????Unknown
bSHostDivNoNan"div_no_nan_1(1      ??9      ??A      ??I      ??a[I~?=_!?iU?p$?????Unknown
yTHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a[I~?=_!?i:0K?????Unknown
?UHostDivNoNan"Egradient_tape/categorical_crossentropy/weighted_loss/value/div_no_nan(1      ??9      ??A      ??I      ??a[I~?=_!?i?%?????Unknown
VHostMul".gradient_tape/sequential/dropout_1/dropout/Mul(1      ??9      ??A      ??I      ??a[I~?=_!?i     ???Unknown
,WHostPow"Adam/Pow(i     ???Unknown
IXHostAssignAddVariableOp"AssignAddVariableOp_1(i     ???Unknown
IYHostAssignAddVariableOp"AssignAddVariableOp_3(i     ???Unknown
'ZHostMul"Mul(i     ???Unknown
J[HostReadVariableOp"div_no_nan_1/ReadVariableOp(i     ???Unknown2CPU