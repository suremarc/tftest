"?V
BHostIDLE"IDLE1    ???@A    ???@a??}???i??}????Unknown
?HostConv2DBackpropFilter":gradient_tape/sequential/conv0/Conv2D/Conv2DBackpropFilter(1     ?@9     ?@A     ?@I     ?@a[;????i?GZ|?????Unknown
oHost_FusedConv2D"sequential/conv0/Relu(1     n?@9     n?@A     n?@I     n?@aV?{!_??i????????Unknown
?HostConv2DBackpropFilter":gradient_tape/sequential/conv1/Conv2D/Conv2DBackpropFilter(1     ?@9     ?@A     ?@I     ?@at?/????i??}?ZA???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1     ??@9     ??@A     ??@I     ??@a]??????i????Z>???Unknown
?HostConv2DBackpropInput"9gradient_tape/sequential/conv1/Conv2D/Conv2DBackpropInput(1     P?@9     P?@A     P?@I     P?@a??G\????i4|*?%???Unknown
mHostMaxPool"sequential/pool0/MaxPool(1     ??@9     ??@A     ??@I     ??@asnt+g??i???cæ???Unknown
?HostMaxPoolGrad"2gradient_tape/sequential/pool0/MaxPool/MaxPoolGrad(1     ?w@9     ?w@A     ?w@I     ?w@a?-???o??iB?ׁ ???Unknown
?	HostBiasAddGrad"2gradient_tape/sequential/conv0/BiasAdd/BiasAddGrad(1     ?m@9     ?m@A     ?m@I     ?m@ağZ]?H|?i??\9???Unknown
o
Host_FusedConv2D"sequential/conv1/Relu(1      j@9      j@A      j@I      j@a,?/?/?x?i?(t?j???Unknown
?HostRandomUniform"9sequential/dropout_1/dropout/random_uniform/RandomUniform(1     ?`@9     ?`@A     ?`@I     ?`@asnt+gp?iebB.????Unknown
?HostMaxPoolGrad"2gradient_tape/sequential/pool1/MaxPool/MaxPoolGrad(1     ?`@9     ?`@A     ?`@I     ?`@a??????o?i??\??????Unknown
mHostMaxPool"sequential/pool1/MaxPool(1      ]@9      ]@A      ]@I      ]@a;??ܯk?it?F??????Unknown
?HostRandomUniform"7sequential/dropout/dropout/random_uniform/RandomUniform(1     ?[@9     ?[@A     ?[@I     ?[@a??	??Aj?i???????Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1     ?W@9     ?W@A     ?W@I     ?W@a?-???of?i:;?M????Unknown
?HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1     ?U@9     ?U@A     ?U@I     ?U@a??Q?ˆd?i3??a????Unknown
}HostReluGrad"'gradient_tape/sequential/conv0/ReluGrad(1     @T@9     @T@A     @T@I     @T@a????HUc?i?*?)???Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1     ?Q@9     ?Q@A     ?Q@I     ?Q@a?5??B?`?i!??0???Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1     @Q@9     @Q@A     @Q@I     @Q@a_??px`?i	?g??@???Unknown
?HostGreaterEqual"'sequential/dropout/dropout/GreaterEqual(1     ?O@9     ?O@A     ?O@I     ?O@a_???^?i
??l?O???Unknown
{HostMatMul"'gradient_tape/sequential/dense/MatMul_1(1      M@9      M@A      M@I      M@a;??ܯ[?iʅ8[u]???Unknown
?HostSoftmaxCrossEntropyWithLogits":categorical_crossentropy/softmax_cross_entropy_with_logits(1      I@9      I@A      I@I      I@ah}:?W?iUD<xdi???Unknown
^HostGatherV2"GatherV2(1      F@9      F@A      F@I      F@a
F??? U?i?+??s???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_4/ResourceApplyAdam(1     ?D@9     ?D@A     ?D@I     ?D@aZ^??b?S?i?n?)?}???Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1      D@9      D@A      D@I      D@a dl.S?i???@:????Unknown?
qHostMul" sequential/dropout_1/dropout/Mul(1      B@9      B@A      B@I      B@a6?&?\/Q?i4C?я???Unknown
`HostGatherV2"
GatherV2_1(1     ?@@9     ?@@A     ?@@I     ?@@a?q??O?i?P6O?????Unknown
qHostMul" sequential/dropout/dropout/Mul_1(1     ?@@9     ?@@A     ?@@I     ?@@a?q??O?im)??????Unknown
}HostReluGrad"'gradient_tape/sequential/conv1/ReluGrad(1      ?@9      ?@A      ?@I      ?@a%?4[??M?i?:???????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      >@9      >@A      =@I      =@a;??ܯK?i????????Unknown
sHostCast"!sequential/dropout_1/dropout/Cast(1      8@9      8@A      8@I      8@a?z?N??F?i?oNF?????Unknown
v HostReadVariableOp"Adam/Cast_3/ReadVariableOp(1      7@9      7@A      7@I      7@a???h?E?i??s?????Unknown
r!Host_FusedMatMul"sequential/dense/BiasAdd(1      7@9      7@A      7@I      7@a???h?E?i?????????Unknown
?"HostGreaterEqual")sequential/dropout_1/dropout/GreaterEqual(1      5@9      5@A      5@I      5@a??%?D?iY?b ?????Unknown
o#HostMul"sequential/dropout/dropout/Mul(1      4@9      4@A      4@I      4@a dl.C?i]??+c????Unknown
?$HostBiasAddGrad"2gradient_tape/sequential/conv1/BiasAdd/BiasAddGrad(1      1@9      1@A      1@I      1@a?A?B?:@?im??q????Unknown
g%HostStridedSlice"strided_slice(1      1@9      1@A      1@I      1@a?A?B?:@?i}-??????Unknown
?&HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1      (@9      (@A      (@I      (@a?z?N??6?iL	I?]????Unknown
?'HostResourceApplyAdam"$Adam/Adam/update_5/ResourceApplyAdam(1      (@9      (@A      (@I      (@a?z?N??6?i?r;????Unknown
i(HostWriteSummary"WriteSummary(1      &@9      &@A      &@I      &@a
F??? 5?iD?n:?????Unknown?
m)HostSoftmax"sequential/dense/Softmax(1      $@9      $@A      $@I      $@a dl.3?i?%<@>????Unknown
q*HostCast"sequential/dropout/dropout/Cast(1      $@9      $@A      $@I      $@a dl.3?iH?	F?????Unknown
t+HostReadVariableOp"Adam/Cast/ReadVariableOp(1      "@9      "@A      "@I      "@a6?&?\/1?i$?1?????Unknown
Z,HostArgMax"ArgMax(1       @9       @A       @I       @a?N??.?iYT?????Unknown
?-HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1      @9      @A      @I      @a??X1t?*?i?i]?[????Unknown
[.HostAddV2"Adam/add(1      @9      @A      @I      @a??X1t?*?iu?q????Unknown
d/HostDataset"Iterator::Model(1     @Y@9     @Y@A      @I      @a??X1t?*?i??(?????Unknown
?0HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?z?N??&?i????!????Unknown
v1HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a dl.#?i,I?HS????Unknown
l2HostIteratorGetNext"IteratorGetNext(1      @9      @A      @I      @a dl.#?im?˄????Unknown
?3HostMul"Lgradient_tape/categorical_crossentropy/softmax_cross_entropy_with_logits/mul(1      @9      @A      @I      @a dl.#?i?լN?????Unknown
~4HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1      @9      @A      @I      @a?N???iHte??????Unknown
\5HostArgMax"ArgMax_1(1      @9      @A      @I      @a?N???i? ?????Unknown
V6HostCast"Cast(1      @9      @A      @I      @a?N???i|?ֈ?????Unknown
?7HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @a?N???iP???????Unknown
e8Host
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @a?N???i??GZ|????Unknown?
s9HostMul""sequential/dropout_1/dropout/Mul_1(1      @9      @A      @I      @a?N???iJ? ?p????Unknown
]:HostCast"Adam/Cast_1(1      @9      @A      @I      @a?z?N???i>?(????Unknown
v;HostReadVariableOp"Adam/Cast_2/ReadVariableOp(1      @9      @A      @I      @a?z?N???i2{`?????Unknown
[<HostPow"
Adam/Pow_1(1      @9      @A      @I      @a?z?N???i&򟮖????Unknown
o=HostReadVariableOp"Adam/ReadVariableOp(1      @9      @A      @I      @a?z?N???ii*?M????Unknown
t>HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a?z?N???i??K????Unknown
V?HostSum"Sum_2(1      @9      @A      @I      @a?z?N???iW???????Unknown
{@HostSum"*categorical_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a?z?N???i????s????Unknown
uAHostReadVariableOp"div_no_nan/ReadVariableOp(1      @9      @A      @I      @a?z?N???i?DT7+????Unknown
?BHostReadVariableOp"&sequential/conv1/Conv2D/ReadVariableOp(1      @9      @A      @I      @a?z?N???i޻ޅ?????Unknown
YCHostPow"Adam/Pow(1       @9       @A       @I       @a?N???i+;?\????Unknown
XDHostEqual"Equal(1       @9       @A       @I       @a?N???ixZ???????Unknown
`EHostDivNoNan"
div_no_nan(1       @9       @A       @I       @a?N???iũ?"Q????Unknown
bFHostDivNoNan"div_no_nan_1(1       @9       @A       @I       @a?N???i?OW?????Unknown
?GHostTile";gradient_tape/categorical_crossentropy/weighted_loss/Tile_1(1       @9       @A       @I       @a?N???i_H??E????Unknown
HHostMul".gradient_tape/sequential/dropout_1/dropout/Mul(1       @9       @A       @I       @a?N???i????????Unknown
?IHostMul"0gradient_tape/sequential/dropout_1/dropout/Mul_1(1       @9       @A       @I       @a?N???i??d?9????Unknown
?JHostReadVariableOp"'sequential/conv0/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a?N???iF6?(?????Unknown
?KHostReadVariableOp"'sequential/conv1/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a?N???i??].????Unknown
?LHostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a?N???i??y??????Unknown
?MHostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1       @9       @A       @I       @a?N???i-$??"????Unknown
vNHostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??a?N???>i?K?_????Unknown
vOHostAssignAddVariableOp"AssignAddVariableOp_4(1      ??9      ??A      ??I      ??a?N???>i{s2??????Unknown
XPHostCast"Cast_1(1      ??9      ??A      ??I      ??a?N???>i"?`?????Unknown
XQHostCast"Cast_2(1      ??9      ??A      ??I      ??a?N???>i?.????Unknown
?RHostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      ??9      ??A      ??I      ??a?N???>ip??HT????Unknown
TSHostMul"Mul(1      ??9      ??A      ??I      ??a?N???>i?b?????Unknown
?THostCast"8categorical_crossentropy/weighted_loss/num_elements/Cast(1      ??9      ??A      ??I      ??a?N???>i?9}?????Unknown
?UHostDivNoNan",categorical_crossentropy/weighted_loss/value(1      ??9      ??A      ??I      ??a?N???>ieaG?????Unknown
wVHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a?N???>i?u?H????Unknown
wWHostReadVariableOp"div_no_nan_1/ReadVariableOp(1      ??9      ??A      ??I      ??a?N???>i???˅????Unknown
?XHostDivNoNan"Egradient_tape/categorical_crossentropy/weighted_loss/value/div_no_nan(1      ??9      ??A      ??I      ??a?N???>iZ????????Unknown
?YHostReadVariableOp"&sequential/conv0/Conv2D/ReadVariableOp(1      ??9      ??A      ??I      ??a?N???>i      ???Unknown
IZHostAssignAddVariableOp"AssignAddVariableOp_3(i      ???Unknown
4[HostIdentity"Identity(i      ???Unknown?
L\HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(i      ???Unknown*?U
?HostConv2DBackpropFilter":gradient_tape/sequential/conv0/Conv2D/Conv2DBackpropFilter(1     ?@9     ?@A     ?@I     ?@aRbj??}??iRbj??}???Unknown
oHost_FusedConv2D"sequential/conv0/Relu(1     n?@9     n?@A     n?@I     n?@a?P3(m??i??Nef????Unknown
?HostConv2DBackpropFilter":gradient_tape/sequential/conv1/Conv2D/Conv2DBackpropFilter(1     ?@9     ?@A     ?@I     ?@a?ÂO???i?0X?n???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1     ??@9     ??@A     ??@I     ??@a?"S????iW??39B???Unknown
?HostConv2DBackpropInput"9gradient_tape/sequential/conv1/Conv2D/Conv2DBackpropInput(1     P?@9     P?@A     P?@I     P?@a?٧V+???i??w?>????Unknown
mHostMaxPool"sequential/pool0/MaxPool(1     ??@9     ??@A     ??@I     ??@al???iV8??H???Unknown
?HostMaxPoolGrad"2gradient_tape/sequential/pool0/MaxPool/MaxPoolGrad(1     ?w@9     ?w@A     ?w@I     ?w@am#?l?	??i?6iI???Unknown
?HostBiasAddGrad"2gradient_tape/sequential/conv0/BiasAdd/BiasAddGrad(1     ?m@9     ?m@A     ?m@I     ?m@a!#??8??i??e?)????Unknown
o	Host_FusedConv2D"sequential/conv1/Relu(1      j@9      j@A      j@I      j@aL#???ԑ?i???y???Unknown
?
HostRandomUniform"9sequential/dropout_1/dropout/random_uniform/RandomUniform(1     ?`@9     ?`@A     ?`@I     ?`@al???ip)C??????Unknown
?HostMaxPoolGrad"2gradient_tape/sequential/pool1/MaxPool/MaxPoolGrad(1     ?`@9     ?`@A     ?`@I     ?`@aa?Ch???ij9T??0???Unknown
mHostMaxPool"sequential/pool1/MaxPool(1      ]@9      ]@A      ]@I      ]@a?F??ʃ?i?T]?????Unknown
?HostRandomUniform"7sequential/dropout/dropout/random_uniform/RandomUniform(1     ?[@9     ?[@A     ?[@I     ?[@a?????Ă?iG	?????Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1     ?W@9     ?W@A     ?W@I     ?W@am#?l?	??i?˼????Unknown
?HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1     ?U@9     ?U@A     ?U@I     ?U@aU٦??X}?iW?y?E???Unknown
}HostReluGrad"'gradient_tape/sequential/conv0/ReluGrad(1     @T@9     @T@A     @T@I     @T@a???t?{?i??ݶ}???Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1     ?Q@9     ?Q@A     ?Q@I     ?Q@a?k3'?:x?i?+,Ë????Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1     @Q@9     @Q@A     @Q@I     @Q@a???΋w?i??[_?????Unknown
?HostGreaterEqual"'sequential/dropout/dropout/GreaterEqual(1     ?O@9     ?O@A     ?O@I     ?O@aw???u?i??-?????Unknown
{HostMatMul"'gradient_tape/sequential/dense/MatMul_1(1      M@9      M@A      M@I      M@a?F???s?ikB?^8/???Unknown
?HostSoftmaxCrossEntropyWithLogits":categorical_crossentropy/softmax_cross_entropy_with_logits(1      I@9      I@A      I@I      I@al+??q?iD??QXQ???Unknown
^HostGatherV2"GatherV2(1      F@9      F@A      F@I      F@a??-??n?i??e?_o???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_4/ResourceApplyAdam(1     ?D@9     ?D@A     ?D@I     ?D@a?"?|z?k?i`?n[????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1      D@9      D@A      D@I      D@a1Gm?Lk?icrO1?????Unknown?
qHostMul" sequential/dropout_1/dropout/Mul(1      B@9      B@A      B@I      B@a???.??h?i=i~:????Unknown
`HostGatherV2"
GatherV2_1(1     ?@@9     ?@@A     ?@@I     ?@@a?Gb ??f?i??~Ϳ????Unknown
qHostMul" sequential/dropout/dropout/Mul_1(1     ?@@9     ?@@A     ?@@I     ?@@a?Gb ??f?i?-?E????Unknown
}HostReluGrad"'gradient_tape/sequential/conv1/ReluGrad(1      ?@9      ?@A      ?@I      ?@aƐT?I(e?i^?`?m???Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      >@9      >@A      =@I      =@a?F???c?i8?"?8???Unknown
sHostCast"!sequential/dropout_1/dropout/Cast(1      8@9      8@A      8@I      8@a??tAa`?i?m???%???Unknown
vHostReadVariableOp"Adam/Cast_3/ReadVariableOp(1      7@9      7@A      7@I      7@ayk;?e_?i??uL5???Unknown
r Host_FusedMatMul"sequential/dense/BiasAdd(1      7@9      7@A      7@I      7@ayk;?e_?i5?a??D???Unknown
?!HostGreaterEqual")sequential/dropout_1/dropout/GreaterEqual(1      5@9      5@A      5@I      5@a???2?\?i4??TS???Unknown
o"HostMul"sequential/dropout/dropout/Mul(1      4@9      4@A      4@I      4@a1Gm?L[?iXB?y?`???Unknown
?#HostBiasAddGrad"2gradient_tape/sequential/conv1/BiasAdd/BiasAddGrad(1      1@9      1@A      1@I      1@a?"?r4W?i?6沔l???Unknown
g$HostStridedSlice"strided_slice(1      1@9      1@A      1@I      1@a?"?r4W?iz+??.x???Unknown
?%HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1      (@9      (@A      (@I      (@a??tAaP?i?}??_????Unknown
?&HostResourceApplyAdam"$Adam/Adam/update_5/ResourceApplyAdam(1      (@9      (@A      (@I      (@a??tAaP?i?b-?????Unknown
i'HostWriteSummary"WriteSummary(1      &@9      &@A      &@I      &@a??-??N?iy?????Unknown?
m(HostSoftmax"sequential/dense/Softmax(1      $@9      $@A      $@I      $@a1Gm?LK?i??F?????Unknown
q)HostCast"sequential/dropout/dropout/Cast(1      $@9      $@A      $@I      $@a1Gm?LK?i?$Dw?????Unknown
t*HostReadVariableOp"Adam/Cast/ReadVariableOp(1      "@9      "@A      "@I      "@a???.??H?iS???ܣ???Unknown
Z+HostArgMax"ArgMax(1       @9       @A       @I       @a'l???E?i.L?R????Unknown
?,HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1      @9      @A      @I      @a????!C?i.ɸ?????Unknown
[-HostAddV2"Adam/add(1      @9      @A      @I      @a????!C?i.y%??????Unknown
d.HostDataset"Iterator::Model(1     @Y@9     @Y@A      @I      @a????!C?i.)?ɧ????Unknown
?/HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a??tAa@?iRR??????Unknown
v0HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a1Gm?L;?i??<?)????Unknown
l1HostIteratorGetNext"IteratorGetNext(1      @9      @A      @I      @a1Gm?L;?i䖊J?????Unknown
?2HostMul"Lgradient_tape/categorical_crossentropy/softmax_cross_entropy_with_logits/mul(1      @9      @A      @I      @a1Gm?L;?i-9???????Unknown
~3HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1      @9      @A      @I      @a'l???5?i?T÷????Unknown
\4HostArgMax"ArgMax_1(1      @9      @A      @I      @a'l???5?i	pT?r????Unknown
V5HostCast"Cast(1      @9      @A      @I      @a'l???5?iw???-????Unknown
?6HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @a'l???5?i???c?????Unknown
e7Host
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @a'l???5?iS?D?????Unknown?
s8HostMul""sequential/dropout_1/dropout/Mul_1(1      @9      @A      @I      @a'l???5?i??L$^????Unknown
]9HostCast"Adam/Cast_1(1      @9      @A      @I      @a??tAa0?iSr{Lj????Unknown
v:HostReadVariableOp"Adam/Cast_2/ReadVariableOp(1      @9      @A      @I      @a??tAa0?i??tv????Unknown
[;HostPow"
Adam/Pow_1(1      @9      @A      @I      @a??tAa0?iw?؜?????Unknown
o<HostReadVariableOp"Adam/ReadVariableOp(1      @9      @A      @I      @a??tAa0?i	0Ŏ????Unknown
t=HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a??tAa0?i??5??????Unknown
V>HostSum"Sum_2(1      @9      @A      @I      @a??tAa0?i-Yd?????Unknown
{?HostSum"*categorical_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a??tAa0?i???=?????Unknown
u@HostReadVariableOp"div_no_nan/ReadVariableOp(1      @9      @A      @I      @a??tAa0?iQ??e?????Unknown
?AHostReadVariableOp"&sequential/conv1/Conv2D/ReadVariableOp(1      @9      @A      @I      @a??tAa0?i????????Unknown
YBHostPow"Adam/Pow(1       @9       @A       @I       @a'l???%?i?$?(????Unknown
XCHostEqual"Equal(1       @9       @A       @I       @a'l???%?iQ2.n?????Unknown
`DHostDivNoNan"
div_no_nan(1       @9       @A       @I       @a'l???%?i@M??????Unknown
bEHostDivNoNan"div_no_nan_1(1       @9       @A       @I       @a'l???%?i?MlNA????Unknown
?FHostTile";gradient_tape/categorical_crossentropy/weighted_loss/Tile_1(1       @9       @A       @I       @a'l???%?iv[???????Unknown
GHostMul".gradient_tape/sequential/dropout_1/dropout/Mul(1       @9       @A       @I       @a'l???%?i-i?.?????Unknown
?HHostMul"0gradient_tape/sequential/dropout_1/dropout/Mul_1(1       @9       @A       @I       @a'l???%?i?vɞY????Unknown
?IHostReadVariableOp"'sequential/conv0/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a'l???%?i????????Unknown
?JHostReadVariableOp"'sequential/conv1/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a'l???%?iR?????Unknown
?KHostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a'l???%?i	?&?q????Unknown
?LHostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1       @9       @A       @I       @a'l???%?i??E_?????Unknown
vMHostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??a'l????i?4U~????Unknown
vNHostAssignAddVariableOp"AssignAddVariableOp_4(1      ??9      ??A      ??I      ??a'l????iv?d?,????Unknown
XOHostCast"Cast_1(1      ??9      ??A      ??I      ??a'l????iQBt??????Unknown
XPHostCast"Cast_2(1      ??9      ??A      ??I      ??a'l????i,Ƀ??????Unknown
?QHostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      ??9      ??A      ??I      ??a'l????iP??8????Unknown
TRHostMul"Mul(1      ??9      ??A      ??I      ??a'l????i?֢??????Unknown
?SHostCast"8categorical_crossentropy/weighted_loss/num_elements/Cast(1      ??9      ??A      ??I      ??a'l????i?]?g?????Unknown
?THostDivNoNan",categorical_crossentropy/weighted_loss/value(1      ??9      ??A      ??I      ??a'l????i???E????Unknown
wUHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a'l????isk???????Unknown
wVHostReadVariableOp"div_no_nan_1/ReadVariableOp(1      ??9      ??A      ??I      ??a'l????iN????????Unknown
?WHostDivNoNan"Egradient_tape/categorical_crossentropy/weighted_loss/value/div_no_nan(1      ??9      ??A      ??I      ??a'l????i)y?GQ????Unknown
?XHostReadVariableOp"&sequential/conv0/Conv2D/ReadVariableOp(1      ??9      ??A      ??I      ??a'l????i     ???Unknown
IYHostAssignAddVariableOp"AssignAddVariableOp_3(i     ???Unknown
4ZHostIdentity"Identity(i     ???Unknown?
L[HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(i     ???Unknown2CPU