"?]
BHostIDLE"IDLE1     w?@A     w?@a?|?]}~??i?|?]}~???Unknown
?HostConv2DBackpropFilter":gradient_tape/sequential/conv0/Conv2D/Conv2DBackpropFilter(1     x?@9     x?@A     x?@I     x?@a?w????i`}?????Unknown
mHostBiasAdd"sequential/dense/BiasAdd(1     ė@9     ė@A     ė@I     ė@a߻?~Ġ?i??l\????Unknown
oHost_FusedConv2D"sequential/conv0/Relu(1     ?@9     ?@A     ?@I     ?@a?!???i?]S????Unknown
?HostRandomUniform"9sequential/dropout_1/dropout/random_uniform/RandomUniform(1     ??@9     ??@A     ??@I     ??@a??"?U_??i???ND???Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1     @?@9     @?@A     @?@I     @?@a???9??i>???????Unknown
?HostMaxPoolGrad"2gradient_tape/sequential/pool0/MaxPool/MaxPoolGrad(1      ?@9      ?@A      ?@I      ?@aE_??n??i8G?͑q???Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1     p?@9     p?@A     p?@I     p?@aa?D?=??iA????Unknown
q	HostMul" sequential/dropout_1/dropout/Mul(1     ??@9     ??@A     ??@I     ??@a?X?(???i???Q????Unknown
?
HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1     p}@9     p}@A     p}@I     p}@a=R???Ą?iO?M?e????Unknown
gHostRelu"sequential/dense/Relu(1     ?|@9     ?|@A     ?|@I     ?|@a?T??i??õ$???Unknown
?HostGreaterEqual")sequential/dropout_1/dropout/GreaterEqual(1     ?w@9     ?w@A     ?w@I     ?w@a??3????i???"bg???Unknown
?HostMatMul"8gradient_tape/sequential/dense/Tensordot/MatMul/MatMul_1(1     ?p@9     ?p@A     ?p@I     ?p@a?۷???w?ir%z?????Unknown
mHostMaxPool"sequential/pool0/MaxPool(1     ?m@9     ?m@A     ?m@I     ?m@a?[??8?t?i)?F?????Unknown
uHostMatMul"!sequential/dense/Tensordot/MatMul(1      m@9      m@A      m@I      m@a+#_}?t?ioVA?????Unknown
?HostMul"0gradient_tape/sequential/dropout_1/dropout/Mul_1(1     @l@9     @l@A     @l@I     @l@a]??"s?s?i?oU'????Unknown
tHost_FusedMatMul"sequential/dense_1/BiasAdd(1     ?j@9     ?j@A     ?j@I     ?j@a>?A??r?iN??Z?7???Unknown
?HostMatMul"6gradient_tape/sequential/dense/Tensordot/MatMul/MatMul(1     `f@9     `f@A     `f@I     `f@a?TB??o?i??W???Unknown
sHostMul""sequential/dropout_1/dropout/Mul_1(1     `f@9     `f@A     `f@I     `f@a?TB??o?i?D??v???Unknown
^HostGatherV2"GatherV2(1     @f@9     @f@A     @f@I     @f@a$/ҹ|eo?i'??????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_4/ResourceApplyAdam(1     ?e@9     ?e@A     ?e@I     ?e@a?r???n?i??ג????Unknown
?HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1      e@9      e@A      e@I      e@a??xg??m?iQ v?4????Unknown
}HostReluGrad"'gradient_tape/sequential/conv0/ReluGrad(1     ?d@9     ?d@A     ?d@I     ?d@a?E??zm?i??DCO????Unknown
}HostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1     ?a@9     ?a@A     ?a@I     ?a@a	?9i?i?]^X????Unknown
?HostRandomUniform"7sequential/dropout/dropout/random_uniform/RandomUniform(1     `a@9     `a@A     `a@I     `a@ayr??v?h?iW?!???Unknown
sHostCast"!sequential/dropout_1/dropout/Cast(1     ?`@9     ?`@A     ?`@I     ?`@a???ǉug?i?EY?8???Unknown
?HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1     ?_@9     ?_@A     ?_@I     ?_@a??֕?ff?iR???N???Unknown
?HostBiasAddGrad"2gradient_tape/sequential/conv0/BiasAdd/BiasAddGrad(1      ^@9      ^@A      ^@I      ^@a??1܇*e?i?M?}d???Unknown
HostMul".gradient_tape/sequential/dropout_1/dropout/Mul(1     @Z@9     @Z@A     @Z@I     @Z@ar??6?b?ik?p??v???Unknown
?HostGreaterEqual"'sequential/dropout/dropout/GreaterEqual(1     ?S@9     ?S@A     ?S@I     ?S@a?=f?[?i
?{燄???Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1     ?S@9     ?S@A     @Q@I     @Q@a?LlpOWX?i0?3??????Unknown
s HostDataset"Iterator::Model::ParallelMapV2(1      P@9      P@A      P@I      P@a<?ēV?i??Bq?????Unknown
u!HostFlushSummaryWriter"FlushSummaryWriter(1      G@9      G@A      G@I      G@a????4:P?i^???????Unknown?
?"HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      E@9      E@A     ?D@I     ?D@a??FS?L?if9??U????Unknown
?#HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1     ?C@9     ?C@A     ?C@I     ?C@a????K?i#cP?6????Unknown
q$HostMul" sequential/dropout/dropout/Mul_1(1      B@9      B@A      B@I      B@a?.ա<fI?io?xu?????Unknown
?%HostReadVariableOp"'sequential/conv0/BiasAdd/ReadVariableOp(1      ?@9      ?@A      ?@I      ?@a?="?%?E?i? ?>????Unknown
q&HostProd"sequential/dense/Tensordot/Prod(1      :@9      :@A      :@I      :@aqLoXXB?i?<?B?????Unknown
?'HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1      9@9      9@A      9@I      9@aϵ~7q?A?i~????Unknown
?(HostMul"Lgradient_tape/categorical_crossentropy/softmax_cross_entropy_with_logits/mul(1      7@9      7@A      7@I      7@a????4:@?i??Y?????Unknown
?)HostSoftmaxCrossEntropyWithLogits":categorical_crossentropy/softmax_cross_entropy_with_logits(1      6@9      6@A      6@I      6@a??Y?-??i??????Unknown
Z*HostArgMax"ArgMax(1      5@9      5@A      5@I      5@a??xg??=?i3?;P?????Unknown
g+HostStridedSlice"strided_slice(1      3@9      3@A      3@I      3@a\??x?:?i?X?????Unknown
b,HostDivNoNan"div_no_nan_1(1      2@9      2@A      2@I      2@a?.ա<f9?i?O?2????Unknown
?-HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1      0@9      0@A      0@I      0@a<?ē6?i p????Unknown
i.HostWriteSummary"WriteSummary(1      .@9      .@A      .@I      .@a??1܇*5?i5?kЩ????Unknown?
d/HostDataset"Iterator::Model(1     @S@9     @S@A      *@I      *@aqLoXX2?i?V??????Unknown
o0HostSoftmax"sequential/dense_1/Softmax(1      *@9      *@A      *@I      *@aqLoXX2?i	?A??????Unknown
?1HostResourceApplyAdam"$Adam/Adam/update_5/ResourceApplyAdam(1      $@9      $@A      $@I      $@aK??%?8,?i??_????Unknown
`2HostGatherV2"
GatherV2_1(1      $@9      $@A      $@I      $@aK??%?8,?i?f???????Unknown
?3HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      "@9      "@A      "@I      "@a?.ա<f)?iN??N]????Unknown
q4HostCast"sequential/dropout/dropout/Cast(1      "@9      "@A      "@I      "@a?.ա<f)?i??z??????Unknown
l5HostIteratorGetNext"IteratorGetNext(1       @9       @A       @I       @a<?ē&?i΂??\????Unknown
~6HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1      @9      @A      @I      @a?yP?K?#?i?'v?????Unknown
[7HostAddV2"Adam/add(1      @9      @A      @I      @a?yP?K?#?i??/?????Unknown
e8Host
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @a?yP?K?#?i?q?,????Unknown?
v9HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @aK??%?8?i?????????Unknown
o:HostMul"sequential/dropout/dropout/Mul(1      @9      @A      @I      @aK??%?8?i^?;??????Unknown
\;HostArgMax"ArgMax_1(1      @9      @A      @I      @a<?ē?i??\V?????Unknown
{<HostSum"*categorical_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a<?ē?i??}?=????Unknown
y=HostGatherV2"#sequential/dense/Tensordot/GatherV2(1      @9      @A      @I      @a<?ē?i#????????Unknown
{>HostGatherV2"%sequential/dense/Tensordot/GatherV2_1(1      @9      @A      @I      @a<?ē?i???0?????Unknown
t?HostReadVariableOp"Adam/Cast/ReadVariableOp(1      @9      @A      @I      @a-????i+BX?.????Unknown
]@HostCast"Adam/Cast_1(1      @9      @A      @I      @a-????i????????Unknown
YAHostPow"Adam/Pow(1      @9      @A      @I      @a-????i???=????Unknown
?BHostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1     @T@9     @T@A      @I      @a-????i~_"?????Unknown
VCHostSum"Sum_2(1      @9      @A      @I      @a-????i???L????Unknown
wDHostReadVariableOp"div_no_nan_1/ReadVariableOp(1      @9      @A      @I      @a-????i`?S??????Unknown
?EHostTile";gradient_tape/categorical_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a-????i?|?n[????Unknown
?FHostReadVariableOp"&sequential/conv0/Conv2D/ReadVariableOp(1      @9      @A      @I      @a-????iB1???????Unknown
?GHostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a-????i??\j????Unknown
yHHostConcatV2"#sequential/dense/Tensordot/concat_1(1      @9      @A      @I      @a-????i$????????Unknown
rIHostPack" sequential/dense/Tensordot/stack(1      @9      @A      @I      @a-????i?NOIy????Unknown
vJHostReadVariableOp"Adam/Cast_2/ReadVariableOp(1       @9       @A       @I       @a<?ē?i??_??????Unknown
vKHostReadVariableOp"Adam/Cast_3/ReadVariableOp(1       @9       @A       @I       @a<?ē?i+?p?-????Unknown
[LHostPow"
Adam/Pow_1(1       @9       @A       @I       @a<?ē?iv??6?????Unknown
oMHostReadVariableOp"Adam/ReadVariableOp(1       @9       @A       @I       @a<?ē?i?/???????Unknown
tNHostAssignAddVariableOp"AssignAddVariableOp(1       @9       @A       @I       @a<?ē?i???<????Unknown
vOHostAssignAddVariableOp"AssignAddVariableOp_3(1       @9       @A       @I       @a<?ē?iW ?#?????Unknown
vPHostAssignAddVariableOp"AssignAddVariableOp_4(1       @9       @A       @I       @a<?ē?i???r?????Unknown
XQHostCast"Cast_2(1       @9       @A       @I       @a<?ē?i???K????Unknown
?RHostCast"8categorical_crossentropy/weighted_loss/num_elements/Cast(1       @9       @A       @I       @a<?ē?i8???????Unknown
uSHostReadVariableOp"div_no_nan/ReadVariableOp(1       @9       @A       @I       @a<?ē?i??_ ????Unknown
?THostDivNoNan"Egradient_tape/categorical_crossentropy/weighted_loss/value/div_no_nan(1       @9       @A       @I       @a<?ē?i?y?Z????Unknown
?UHostReadVariableOp")sequential/dense/Tensordot/ReadVariableOp(1       @9       @A       @I       @a<?ē?i???????Unknown
?VHostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a<?ē?idj%M????Unknown
vWHostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??a<?ē?>i???t<????Unknown
VXHostCast"Cast(1      ??9      ??A      ??I      ??a<?ē?>i??5?i????Unknown
XYHostCast"Cast_1(1      ??9      ??A      ??I      ??a<?ē?>i??Ö????Unknown
XZHostEqual"Equal(1      ??9      ??A      ??I      ??a<?ē?>i?ZF??????Unknown
a[HostIdentity"Identity(1      ??9      ??A      ??I      ??a<?ē?>i"???????Unknown?
?\HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      ??9      ??A      ??I      ??a<?ē?>iH?V:????Unknown
?]HostDivNoNan",categorical_crossentropy/weighted_loss/value(1      ??9      ??A      ??I      ??a<?ē?>in?aK????Unknown
`^HostDivNoNan"
div_no_nan(1      ??9      ??A      ??I      ??a<?ē?>i?Kg?x????Unknown
y_HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a<?ē?>i??ﰥ????Unknown
s`HostProd"!sequential/dense/Tensordot/Prod_1(1      ??9      ??A      ??I      ??a<?ē?>i??w??????Unknown
?aHostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??a<?ē?>i     ???Unknown
'bHostMul"Mul(i     ???Unknown
JcHostReadVariableOp"div_no_nan/ReadVariableOp_1(i     ???Unknown*?\
?HostConv2DBackpropFilter":gradient_tape/sequential/conv0/Conv2D/Conv2DBackpropFilter(1     x?@9     x?@A     x?@I     x?@a??0??p??i??0??p???Unknown
mHostBiasAdd"sequential/dense/BiasAdd(1     ė@9     ė@A     ė@I     ė@a???D???iQ?m?????Unknown
oHost_FusedConv2D"sequential/conv0/Relu(1     ?@9     ?@A     ?@I     ?@ah??^??iq???;2???Unknown
?HostRandomUniform"9sequential/dropout_1/dropout/random_uniform/RandomUniform(1     ??@9     ??@A     ??@I     ??@aR>?3???i;??????Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1     @?@9     @?@A     @?@I     @?@a?8?G??ih??^???Unknown
?HostMaxPoolGrad"2gradient_tape/sequential/pool0/MaxPool/MaxPoolGrad(1      ?@9      ?@A      ?@I      ?@ao?;D??i6??)???Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1     p?@9     p?@A     p?@I     p?@a{o??B??i.??hJ????Unknown
qHostMul" sequential/dropout_1/dropout/Mul(1     ??@9     ??@A     ??@I     ??@aͭ?'?H??i??+?W???Unknown
?	HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1     p}@9     p}@A     p}@I     p}@a???q???i??F??T???Unknown
g
HostRelu"sequential/dense/Relu(1     ?|@9     ?|@A     ?|@I     ?|@a?BH????i?,???L???Unknown
?HostGreaterEqual")sequential/dropout_1/dropout/GreaterEqual(1     ?w@9     ?w@A     ?w@I     ?w@a[????b??i?ŗ????Unknown
?HostMatMul"8gradient_tape/sequential/dense/Tensordot/MatMul/MatMul_1(1     ?p@9     ?p@A     ?p@I     ?p@a?w???!??i??^?????Unknown
mHostMaxPool"sequential/pool0/MaxPool(1     ?m@9     ?m@A     ?m@I     ?m@aWxɢ???i!?*?|'???Unknown
uHostMatMul"!sequential/dense/Tensordot/MatMul(1      m@9      m@A      m@I      m@a!-?{K??i??ت????Unknown
?HostMul"0gradient_tape/sequential/dropout_1/dropout/Mul_1(1     @l@9     @l@A     @l@I     @l@aMi8??Z??i{?????Unknown
tHost_FusedMatMul"sequential/dense_1/BiasAdd(1     ?j@9     ?j@A     ?j@I     ?j@a???!????i?[?O?????Unknown
?HostMatMul"6gradient_tape/sequential/dense/Tensordot/MatMul/MatMul(1     `f@9     `f@A     `f@I     `f@aP?Y??
??i??hJ?????Unknown
sHostMul""sequential/dropout_1/dropout/Mul_1(1     `f@9     `f@A     `f@I     `f@aP?Y??
??i?+NE?Q???Unknown
^HostGatherV2"GatherV2(1     @f@9     @f@A     @f@I     @f@a?:h\???i?ﶏ????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_4/ResourceApplyAdam(1     ?e@9     ?e@A     ?e@I     ?e@a?0??p<??i`|9z????Unknown
?HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1      e@9      e@A      e@I      e@a?[?<????i?e-??h???Unknown
}HostReluGrad"'gradient_tape/sequential/conv0/ReluGrad(1     ?d@9     ?d@A     ?d@I     ?d@a?u?I^)??i??Si????Unknown
}HostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1     ?a@9     ?a@A     ?a@I     ?a@a?2?P?4??iq<??<???Unknown
?HostRandomUniform"7sequential/dropout/dropout/random_uniform/RandomUniform(1     `a@9     `a@A     `a@I     `a@a'U[b???iƩ?=?X???Unknown
sHostCast"!sequential/dropout_1/dropout/Cast(1     ?`@9     ?`@A     ?`@I     ?`@a???%݁?i?_?^????Unknown
?HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1     ?_@9     ?_@A     ?_@I     ?_@aN??>???iچZ??????Unknown
?HostBiasAddGrad"2gradient_tape/sequential/conv0/BiasAdd/BiasAddGrad(1      ^@9      ^@A      ^@I      ^@az????i?vv?%???Unknown
HostMul".gradient_tape/sequential/dropout_1/dropout/Mul(1     @Z@9     @Z@A     @Z@I     @Z@a??8??4|?i???L{]???Unknown
?HostGreaterEqual"'sequential/dropout/dropout/GreaterEqual(1     ?S@9     ?S@A     ?S@I     ?S@aԱ??8u?i^???????Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1     ?S@9     ?S@A     @Q@I     @Q@a?];???r?i??(??????Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1      P@9      P@A      P@I      P@a????(1q?i)?H?`????Unknown
u HostFlushSummaryWriter"FlushSummaryWriter(1      G@9      G@A      G@I      G@aU??N??h?i?ŗ?????Unknown?
?!HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      E@9      E@A     ?D@I     ?D@aA~z??f?iy@??????Unknown
?"HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1     ?C@9     ?C@A     ?C@I     ?C@a?zo??d?i<??????Unknown
q#HostMul" sequential/dropout/dropout/Mul_1(1      B@9      B@A      B@I      B@a-*??MWc?if???i&???Unknown
?$HostReadVariableOp"'sequential/conv0/BiasAdd/ReadVariableOp(1      ?@9      ?@A      ?@I      ?@a?{K??`?i<2?u7???Unknown
q%HostProd"sequential/dense/Tensordot/Prod(1      :@9      :@A      :@I      :@a????[?i?.?f	E???Unknown
?&HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1      9@9      9@A      9@I      9@a?H?`??Z?ib???wR???Unknown
?'HostMul"Lgradient_tape/categorical_crossentropy/softmax_cross_entropy_with_logits/mul(1      7@9      7@A      7@I      7@aU??N??X?iK(?#?^???Unknown
?(HostSoftmaxCrossEntropyWithLogits":categorical_crossentropy/softmax_cross_entropy_with_logits(1      6@9      6@A      6@I      6@a?ŗ?W?iW%???j???Unknown
Z)HostArgMax"ArgMax(1      5@9      5@A      5@I      5@a?[?<??V?i??;2?u???Unknown
g*HostStridedSlice"strided_slice(1      3@9      3@A      3@I      3@ah??*`jT?i?Qb"????Unknown
b+HostDivNoNan"div_no_nan_1(1      2@9      2@A      2@I      2@a-*??MWS?i?"	Ή???Unknown
?,HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1      0@9      0@A      0@I      0@a????(1Q?igj?f????Unknown
i-HostWriteSummary"WriteSummary(1      .@9      .@A      .@I      .@az??P?ic?m?u????Unknown?
d.HostDataset"Iterator::Model(1     @S@9     @S@A      *@I      *@a????K?i???q????Unknown
o/HostSoftmax"sequential/dense_1/Softmax(1      *@9      *@A      *@I      *@a????K?i??b?m????Unknown
?0HostResourceApplyAdam"$Adam/Adam/update_5/ResourceApplyAdam(1      $@9      $@A      $@I      $@a????r}E?i???̭???Unknown
`1HostGatherV2"
GatherV2_1(1      $@9      $@A      $@I      $@a????r}E?i5??R,????Unknown
?2HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      "@9      "@A      "@I      "@a-*??MWC?i %&????Unknown
q3HostCast"sequential/dropout/dropout/Cast(1      "@9      "@A      "@I      "@a-*??MWC?iː??׼???Unknown
l4HostIteratorGetNext"IteratorGetNext(1       @9       @A       @I       @a????(1A?i???C$????Unknown
~5HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1      @9      @A      @I      @a~z??>?i???????Unknown
[6HostAddV2"Adam/add(1      @9      @A      @I      @a~z??>?i֍pũ????Unknown
e7Host
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @a~z??>?i?P?l????Unknown?
v8HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a????r}5?i9??4????Unknown
o9HostMul"sequential/dropout/dropout/Mul(1      @9      @A      @I      @a????r}5?i????????Unknown
\:HostArgMax"ArgMax_1(1      @9      @A      @I      @a????(11?i?????Unknown
{;HostSum"*categorical_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a????(11?iy
!-????Unknown
y<HostGatherV2"#sequential/dense/Tensordot/GatherV2(1      @9      @A      @I      @a????(11?i?	3R>????Unknown
{=HostGatherV2"%sequential/dense/Tensordot/GatherV2_1(1      @9      @A      @I      @a????(11?ie	Ewd????Unknown
t>HostReadVariableOp"Adam/Cast/ReadVariableOp(1      @9      @A      @I      @a???׼?)?i??????Unknown
]?HostCast"Adam/Cast_1(1      @9      @A      @I      @a???׼?)?i?஝????Unknown
Y@HostPow"Adam/Pow(1      @9      @A      @I      @a???׼?)?i0??J:????Unknown
?AHostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1     @T@9     @T@A      @I      @a???׼?)?i?{??????Unknown
VBHostSum"Sum_2(1      @9      @A      @I      @a???׼?)?ib?H?s????Unknown
wCHostReadVariableOp"div_no_nan_1/ReadVariableOp(1      @9      @A      @I      @a???׼?)?i?????Unknown
?DHostTile";gradient_tape/categorical_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a???׼?)?i??㹬????Unknown
?EHostReadVariableOp"&sequential/conv0/Conv2D/ReadVariableOp(1      @9      @A      @I      @a???׼?)?i-?UI????Unknown
?FHostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a???׼?)?iƅ~??????Unknown
yGHostConcatV2"#sequential/dense/Tensordot/concat_1(1      @9      @A      @I      @a???׼?)?i_L??????Unknown
rHHostPack" sequential/dense/Tensordot/stack(1      @9      @A      @I      @a???׼?)?i??)????Unknown
vIHostReadVariableOp"Adam/Cast_2/ReadVariableOp(1       @9       @A       @I       @a????(1!?i???;2????Unknown
vJHostReadVariableOp"Adam/Cast_3/ReadVariableOp(1       @9       @A       @I       @a????(1!?in?+NE????Unknown
[KHostPow"
Adam/Pow_1(1       @9       @A       @I       @a????(1!?i)??`X????Unknown
oLHostReadVariableOp"Adam/ReadVariableOp(1       @9       @A       @I       @a????(1!?i??=sk????Unknown
tMHostAssignAddVariableOp"AssignAddVariableOp(1       @9       @A       @I       @a????(1!?i??ƅ~????Unknown
vNHostAssignAddVariableOp"AssignAddVariableOp_3(1       @9       @A       @I       @a????(1!?iZ?O??????Unknown
vOHostAssignAddVariableOp"AssignAddVariableOp_4(1       @9       @A       @I       @a????(1!?i?ت?????Unknown
XPHostCast"Cast_2(1       @9       @A       @I       @a????(1!?iЂa??????Unknown
?QHostCast"8categorical_crossentropy/weighted_loss/num_elements/Cast(1       @9       @A       @I       @a????(1!?i?????????Unknown
uRHostReadVariableOp"div_no_nan/ReadVariableOp(1       @9       @A       @I       @a????(1!?iF?s??????Unknown
?SHostDivNoNan"Egradient_tape/categorical_crossentropy/weighted_loss/value/div_no_nan(1       @9       @A       @I       @a????(1!?i????????Unknown
?THostReadVariableOp")sequential/dense/Tensordot/ReadVariableOp(1       @9       @A       @I       @a????(1!?i???????Unknown
?UHostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a????(1!?iw?????Unknown
vVHostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??a????(1?iUS??????Unknown
VWHostCast"Cast(1      ??9      ??A      ??I      ??a????(1?i3??,*????Unknown
XXHostCast"Cast_1(1      ??9      ??A      ??I      ??a????(1?iܵ?????Unknown
XYHostEqual"Equal(1      ??9      ??A      ??I      ??a????(1?i?? ?=????Unknown
aZHostIdentity"Identity(1      ??9      ??A      ??I      ??a????(1?i? e??????Unknown?
?[HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      ??9      ??A      ??I      ??a????(1?i???QP????Unknown
?\HostDivNoNan",categorical_crossentropy/weighted_loss/value(1      ??9      ??A      ??I      ??a????(1?i? ???????Unknown
`]HostDivNoNan"
div_no_nan(1      ??9      ??A      ??I      ??a????(1?ig?2dc????Unknown
y^HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a????(1?iE w??????Unknown
s_HostProd"!sequential/dense/Tensordot/Prod_1(1      ??9      ??A      ??I      ??a????(1?i#??vv????Unknown
?`HostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??a????(1?i      ???Unknown
'aHostMul"Mul(i      ???Unknown
JbHostReadVariableOp"div_no_nan/ReadVariableOp_1(i      ???Unknown2CPU