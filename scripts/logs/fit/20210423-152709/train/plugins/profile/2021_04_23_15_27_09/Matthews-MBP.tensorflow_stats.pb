"?W
BHostIDLE"IDLE1H?z^\?@AH?z^\?@a??kUA???i??kUA????Unknown
oHost_FusedConv2D"sequential/conv0/Relu(19??v>W?@99??v>W?@A9??v>W?@I9??v>W?@a{??=???i8&K	0????Unknown
?HostConv2DBackpropFilter":gradient_tape/sequential/conv0/Conv2D/Conv2DBackpropFilter(1??C?l??@9??C?l??@A??C?l??@I??C?l??@a?????i"??$????Unknown
mHostMaxPool"sequential/pool0/MaxPool(1NbX9`?@9NbX9`?@ANbX9`?@INbX9`?@aҵeʒ??i?9????Unknown
?HostMaxPoolGrad"2gradient_tape/sequential/pool0/MaxPool/MaxPoolGrad(1{?G?,?@9{?G?,?@A{?G?,?@I{?G?,?@aEb???0??i?(?	H????Unknown
?HostConv2DBackpropFilter":gradient_tape/sequential/conv1/Conv2D/Conv2DBackpropFilter(1'1??@9'1??@A'1??@I'1??@a2?a`ǘ?ie???????Unknown
?HostConv2DBackpropInput"9gradient_tape/sequential/conv1/Conv2D/Conv2DBackpropInput(19??v??|@99??v??|@A9??v??|@I9??v??|@a???g???iЃ?@???Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1?n??&v@9?n??&v@A?n??&v@I?n??&v@al???pŇ?iif?nV|???Unknown
{	HostMatMul"'gradient_tape/sequential/dense/MatMul_1(1H?z??p@9H?z??p@AH?z??p@IH?z??p@a?eC{R5??it??+????Unknown
?
HostBiasAddGrad"2gradient_tape/sequential/conv0/BiasAdd/BiasAddGrad(1
ףp=?o@9
ףp=?o@A
ףp=?o@I
ףp=?o@a?Rr?}???iM=ծ????Unknown
oHost_FusedConv2D"sequential/conv1/Relu(1T㥛?Tj@9T㥛?Tj@AT㥛?Tj@IT㥛?Tj@a5?Q??A|?i???iA???Unknown
?HostRandomUniform"9sequential/dropout_1/dropout/random_uniform/RandomUniform(1bX9??f@9bX9??f@AbX9??f@IbX9??f@aRJ?g?x?i?t?qr???Unknown
?HostGreaterEqual")sequential/dropout_1/dropout/GreaterEqual(1'1?nb@9'1?nb@A'1?nb@I'1?nb@aa??"?s?i??e+ ????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_4/ResourceApplyAdam(1/?$?)`@9/?$?)`@A/?$?)`@I/?$?)`@a8'??Xq?i??N?????Unknown
?HostRandomUniform"7sequential/dropout/dropout/random_uniform/RandomUniform(1q=
ף8\@9q=
ף8\@Aq=
ף8\@Iq=
ף8\@a?݂??Hn?i?l_D?????Unknown
rHost_FusedMatMul"sequential/dense/BiasAdd(1??S㥳W@9??S㥳W@A??S㥳W@I??S㥳W@a??\oi?igk?h????Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1sh??|?W@9sh??|?W@Ash??|?W@Ish??|?W@ar????h?i?)?T[???Unknown
^HostGatherV2"GatherV2(1-????T@9-????T@A-????T@I-????T@a?y???Uf?i????#???Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1??S㥓Q@9??S㥓Q@A??K7?!O@I??K7?!O@a?1??)?`?i???e4???Unknown
sHostMul""sequential/dropout_1/dropout/Mul_1(1H?z?7K@9H?z?7K@AH?z?7K@IH?z?7K@aG?Uf55]?i?y??B???Unknown
?HostGreaterEqual"'sequential/dropout/dropout/GreaterEqual(1R???qI@9R???qI@AR???qI@IR???qI@af?S+DN[?i?h?ɦP???Unknown
}HostReluGrad"'gradient_tape/sequential/conv1/ReluGrad(1=
ףpF@9=
ףpF@A=
ףpF@I=
ףpF@asU*?F?W?i???{\???Unknown
qHostMul" sequential/dropout_1/dropout/Mul(1+??ΧE@9+??ΧE@A+??ΧE@I+??ΧE@axe6=W?iS?(?h???Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1????sD@9????sD@A????sD@I????sD@a?s?e??U?i{[?s???Unknown?
?HostBiasAddGrad"2gradient_tape/sequential/conv1/BiasAdd/BiasAddGrad(1?&1?<B@9?&1?<B@A?&1?<B@I?&1?<B@a?o?|?S?iEř ?|???Unknown
?HostSoftmaxCrossEntropyWithLogits":categorical_crossentropy/softmax_cross_entropy_with_logits(1??|?5?<@9??|?5?<@A??|?5?<@I??|?5?<@aG1???O?i?,??????Unknown
qHostMul" sequential/dropout/dropout/Mul_1(1T㥛? <@9T㥛? <@AT㥛? <@IT㥛? <@aK??k N?i|'????Unknown
sHostCast"!sequential/dropout_1/dropout/Cast(1sh??|?:@9sh??|?:@Ash??|?:@Ish??|?:@a*U??L?i????K????Unknown
?HostMul"0gradient_tape/sequential/dropout_1/dropout/Mul_1(1??C??9@9??C??9@A??C??9@I??C??9@a:uւ??K?i\c?n@????Unknown
HostMul".gradient_tape/sequential/dropout_1/dropout/Mul(1?A`???7@9?A`???7@A?A`???7@I?A`???7@a۪???I?i?H??????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1V-???7@9V-???7@Aj?t?$3@Ij?t?$3@aj?nm?D?i?}??˥???Unknown
d HostDataset"Iterator::Model(1j?t??[@9j?t??[@A1?ZD2@I1?ZD2@a<NWR?C?iQ:?????Unknown
m!HostSoftmax"sequential/dense/Softmax(1
ףp=*0@9
ףp=*0@A
ףp=*0@I
ףp=*0@a7^??XA?i?h2T????Unknown
?"HostResourceApplyAdam"$Adam/Adam/update_5/ResourceApplyAdam(1J+?/@9J+?/@AJ+?/@IJ+?/@a?;??A?@?i?e??3????Unknown
`#HostGatherV2"
GatherV2_1(1J+??.@9J+??.@AJ+??.@IJ+??.@aUy????@?i??`?V????Unknown
i$HostWriteSummary"WriteSummary(1Zd;?O?(@9Zd;?O?(@AZd;?O?(@IZd;?O?(@a?d????:?i?)???????Unknown?
?%HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1???x??&@9???x??&@A???x??&@I???x??&@a?Tئ?8?i?4?????Unknown
q&HostCast"sequential/dropout/dropout/Cast(1-????&@9-????&@A-????&@I-????&@a?)?	P8?it[
?????Unknown
?'HostMul"Lgradient_tape/categorical_crossentropy/softmax_cross_entropy_with_logits/mul(1Zd;?OM&@9Zd;?OM&@AZd;?OM&@IZd;?OM&@aG9????7?i?????????Unknown
?(HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1?K7?A &@9?K7?A &@A?K7?A &@I?K7?A &@a????w?7?i~鑳?????Unknown
?)HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1?????&@9?????&@A?????&@I?????&@a?2????7?i?Gq??????Unknown
}*HostReluGrad"'gradient_tape/sequential/conv0/ReluGrad(1q=
ףp%@9q=
ףp%@Aq=
ףp%@Iq=
ףp%@a?7??7?ik??&?????Unknown
?+HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1??S??[$@9??S??[$@A??S??[$@I??S??[$@a?P>1??5?i5?*?M????Unknown
[,HostAddV2"Adam/add(1w??/?#@9w??/?#@Aw??/?#@Iw??/?#@aY?^?Q5?il?`?????Unknown
l-HostIteratorGetNext"IteratorGetNext(1T㥛Ġ#@9T㥛Ġ#@AT㥛Ġ#@IT㥛Ġ#@a???f75?i?I?g?????Unknown
?.HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1ףp=
 @9ףp=
 @Aףp=
 @Iףp=
 @a??=AD1?i?????????Unknown
v/HostAssignAddVariableOp"AssignAddVariableOp_2(1?? ?r?@9?? ?r?@A?? ?r?@I?? ?r?@a{f3??1?i"o??????Unknown
g0HostStridedSlice"strided_slice(1??(\??@9??(\??@A??(\??@I??(\??@a	I???/?i?h??????Unknown
o1HostMul"sequential/dropout/dropout/Mul(11?Zd@91?Zd@A1?Zd@I1?Zd@a n/?%e-?i????????Unknown
x2HostDataset"#Iterator::Model::ParallelMapV2::Zip(1???x??Y@9???x??Y@A???x??@I???x??@a^k??ڙ,?i?m??????Unknown
?3HostReadVariableOp"&sequential/conv0/Conv2D/ReadVariableOp(1sh??|?@9sh??|?@Ash??|?@Ish??|?@a>?nu%+?i @?96????Unknown
t4HostReadVariableOp"Adam/Cast/ReadVariableOp(1}?5^?I@9}?5^?I@A}?5^?I@I}?5^?I@a?:j*?i?#g@?????Unknown
?5HostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1}?5^??@9}?5^??@A}?5^??@I}?5^??@af??eUt(?iH???^????Unknown
Z6HostArgMax"ArgMax(1?A`?Т@9?A`?Т@A?A`?Т@I?A`?Т@a+x?J(?i??.?????Unknown
V7HostSum"Sum_2(1+??N@9+??N@A+??N@I+??N@a!?????&?i???
Q????Unknown
?8HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1X9??v>@9X9??v>@AX9??v>@IX9??v>@aZ??q?%?i????????Unknown
\9HostArgMax"ArgMax_1(1;?O???@9;?O???@A;?O???@I;?O???@a*?B?c%?i???????Unknown
?:HostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1?l????@9?l????@A?l????@I?l????@aC͸NT$?i\??"H????Unknown
?;HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1?G?z?@9?G?z?@A?G?z?@I?G?z?@aU?? ?$?i|?$߈????Unknown
o<HostReadVariableOp"Adam/ReadVariableOp(1??x?&1@9??x?&1@A??x?&1@I??x?&1@a!~???#?i\??:?????Unknown
?=HostTile";gradient_tape/categorical_crossentropy/weighted_loss/Tile_1(1?????K@9?????K@A?????K@I?????K@a	\c??"?i)?3?????Unknown
?>HostReadVariableOp"'sequential/conv0/BiasAdd/ReadVariableOp(1?V-@9?V-@A?V-@I?V-@a?3??n"?i@?U????Unknown
{?HostSum"*categorical_crossentropy/weighted_loss/Sum(1??(\??@9??(\??@A??(\??@I??(\??@a?-jRQ?!?i??j?0????Unknown
e@Host
LogicalAnd"
LogicalAnd(1L7?A`?@9L7?A`?@AL7?A`?@IL7?A`?@a????
 ?i ?̊1????Unknown?
tAHostAssignAddVariableOp"AssignAddVariableOp(1+???@9+???@A+???@I+???@aB?S???i*M?q1????Unknown
`BHostDivNoNan"
div_no_nan(1?O??n@9?O??n@A?O??n@I?O??n@a?Cʐ?2?i|?3+????Unknown
YCHostPow"Adam/Pow(1V-??@9V-??@AV-??@IV-??@a?h^?g??ior? ????Unknown
vDHostAssignAddVariableOp"AssignAddVariableOp_4(1T㥛? 	@9T㥛? 	@AT㥛? 	@IT㥛? 	@aF8?
.??i?b?k?????Unknown
[EHostPow"
Adam/Pow_1(1j?t?@9j?t?@Aj?t?@Ij?t?@a?????i?:I?????Unknown
]FHostCast"Adam/Cast_1(1NbX9?@9NbX9?@ANbX9?@INbX9?@aRm.I?o?itɒ????Unknown
?GHostReadVariableOp"'sequential/conv1/BiasAdd/ReadVariableOp(1V-???@9V-???@AV-???@IV-???@a?Mk???i?-??W????Unknown
uHHostReadVariableOp"div_no_nan/ReadVariableOp(1Zd;?O@9Zd;?O@AZd;?O@IZd;?O@aL??????i???????Unknown
?IHostDivNoNan"Egradient_tape/categorical_crossentropy/weighted_loss/value/div_no_nan(1NbX9?@9NbX9?@ANbX9?@INbX9?@aT ?_?i??:?????Unknown
bJHostDivNoNan"div_no_nan_1(1sh??|?@9sh??|?@Ash??|?@Ish??|?@a*??a??i???MC????Unknown
?KHostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1j?t?R@9j?t?R@AV-?? @IV-?? @a??U???i?????????Unknown
VLHostCast"Cast(1?Zd; @9?Zd; @A?Zd; @I?Zd; @a?t?Ck?iC??M]????Unknown
XMHostEqual"Equal(1?p=
ף??9?p=
ף??A?p=
ף??I?p=
ף??a;R?O?p?i&v???????Unknown
vNHostReadVariableOp"Adam/Cast_3/ReadVariableOp(1D?l?????9D?l?????AD?l?????ID?l?????a??P??ii??(`????Unknown
~OHostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1NbX9???9NbX9???ANbX9???INbX9???aQ?\????i?MR_?????Unknown
?PHostDivNoNan",categorical_crossentropy/weighted_loss/value(1?Zd;??9?Zd;??A?Zd;??I?Zd;??a`Ǧ)19?iw?DP????Unknown
wQHostReadVariableOp"div_no_nan_1/ReadVariableOp(1?O??n??9?O??n??A?O??n??I?O??n??a? ?<?i?`	y?????Unknown
XRHostCast"Cast_2(1????????9????????A????????I????????a?hS	?i?ߩ?)????Unknown
vSHostReadVariableOp"Adam/Cast_2/ReadVariableOp(1?E??????9?E??????A?E??????I?E??????a9???k??i?GYȋ????Unknown
TTHostMul"Mul(1ffffff??9ffffff??Affffff??Iffffff??aV???	?i?Q??????Unknown
vUHostAssignAddVariableOp"AssignAddVariableOp_3(1m???????9m???????Am???????Im???????a~?h??k?iST?I????Unknown
?VHostReadVariableOp"&sequential/conv1/Conv2D/ReadVariableOp(1??Q????9??Q????A??Q????I??Q????a??????i??G??????Unknown
vWHostAssignAddVariableOp"AssignAddVariableOp_1(1P??n???9P??n???AP??n???IP??n???aaN?Ѡ??i???s?????Unknown
?XHostCast"8categorical_crossentropy/weighted_loss/num_elements/Cast(1?MbX9??9?MbX9??A?MbX9??I?MbX9??at"f7???iW?ӭC????Unknown
XYHostCast"Cast_1(1ffffff??9ffffff??Affffff??Iffffff??aCյ??O ?i.N???????Unknown
yZHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1;?O??n??9;?O??n??A;?O??n??I;?O??n??a?@?u??>iu9??????Unknown
w[HostReadVariableOp"div_no_nan/ReadVariableOp_1(1?t?V??9?t?V??A?t?V??I?t?V??a????(??>iK:?????Unknown
a\HostIdentity"Identity(1???x?&??9???x?&??A???x?&??I???x?&??aDQ?????>i     ???Unknown?*?W
oHost_FusedConv2D"sequential/conv0/Relu(19??v>W?@99??v>W?@A9??v>W?@I9??v>W?@a??+l??i??+l???Unknown
?HostConv2DBackpropFilter":gradient_tape/sequential/conv0/Conv2D/Conv2DBackpropFilter(1??C?l??@9??C?l??@A??C?l??@I??C?l??@a$V&p????ip1?5????Unknown
mHostMaxPool"sequential/pool0/MaxPool(1NbX9`?@9NbX9`?@ANbX9`?@INbX9`?@a(??x????i??@쮦???Unknown
?HostMaxPoolGrad"2gradient_tape/sequential/pool0/MaxPool/MaxPoolGrad(1{?G?,?@9{?G?,?@A{?G?,?@I{?G?,?@a??d???i???d???Unknown
?HostConv2DBackpropFilter":gradient_tape/sequential/conv1/Conv2D/Conv2DBackpropFilter(1'1??@9'1??@A'1??@I'1??@a???n?ֵ?im???n???Unknown
?HostConv2DBackpropInput"9gradient_tape/sequential/conv1/Conv2D/Conv2DBackpropInput(19??v??|@99??v??|@A9??v??|@I9??v??|@acR?Y?C??i?OcL?????Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1?n??&v@9?n??&v@A?n??&v@I?n??&v@a??h/Q???i??Y_?"???Unknown
{HostMatMul"'gradient_tape/sequential/dense/MatMul_1(1H?z??p@9H?z??p@AH?z??p@IH?z??p@a{*?0??i??ln?#???Unknown
?	HostBiasAddGrad"2gradient_tape/sequential/conv0/BiasAdd/BiasAddGrad(1
ףp=?o@9
ףp=?o@A
ףp=?o@I
ףp=?o@a&cI؝?i????c???Unknown
o
Host_FusedConv2D"sequential/conv1/Relu(1T㥛?Tj@9T㥛?Tj@AT㥛?Tj@IT㥛?Tj@aP??R???iBuL?????Unknown
?HostRandomUniform"9sequential/dropout_1/dropout/random_uniform/RandomUniform(1bX9??f@9bX9??f@AbX9??f@IbX9??f@aZ?O"????i,??E{????Unknown
?HostGreaterEqual")sequential/dropout_1/dropout/GreaterEqual(1'1?nb@9'1?nb@A'1?nb@I'1?nb@a? k?Rn??i4k?????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_4/ResourceApplyAdam(1/?$?)`@9/?$?)`@A/?$?)`@I/?$?)`@a?Gٺb???iR~Vg7????Unknown
?HostRandomUniform"7sequential/dropout/dropout/random_uniform/RandomUniform(1q=
ף8\@9q=
ף8\@Aq=
ף8\@Iq=
ף8\@a?O?b????i?s?@?????Unknown
rHost_FusedMatMul"sequential/dense/BiasAdd(1??S㥳W@9??S㥳W@A??S㥳W@I??S㥳W@a?OJ?j??iϜW	?P???Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1sh??|?W@9sh??|?W@Ash??|?W@Ish??|?W@a??WQ????im??^?????Unknown
^HostGatherV2"GatherV2(1-????T@9-????T@A-????T@I-????T@aBMi+???i??J?U????Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1??S㥓Q@9??S㥓Q@A??K7?!O@I??K7?!O@a??|yq}?i?YD?82???Unknown
sHostMul""sequential/dropout_1/dropout/Mul_1(1H?z?7K@9H?z?7K@AH?z?7K@IH?z?7K@a??h???y?i?+??e???Unknown
?HostGreaterEqual"'sequential/dropout/dropout/GreaterEqual(1R???qI@9R???qI@AR???qI@IR???qI@aS???x?i??(֕???Unknown
}HostReluGrad"'gradient_tape/sequential/conv1/ReluGrad(1=
ףpF@9=
ףpF@A=
ףpF@I=
ףpF@ao"?`?t?i?BcՌ????Unknown
qHostMul" sequential/dropout_1/dropout/Mul(1+??ΧE@9+??ΧE@A+??ΧE@I+??ΧE@az#?ZA{t?i?XX?????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1????sD@9????sD@A????sD@I????sD@a???Xs?i?ۡ?3???Unknown?
?HostBiasAddGrad"2gradient_tape/sequential/conv1/BiasAdd/BiasAddGrad(1?&1?<B@9?&1?<B@A?&1?<B@I?&1?<B@a:m?c??q?i?bi??1???Unknown
?HostSoftmaxCrossEntropyWithLogits":categorical_crossentropy/softmax_cross_entropy_with_logits(1??|?5?<@9??|?5?<@A??|?5?<@I??|?5?<@a????kk?iZA2?M???Unknown
qHostMul" sequential/dropout/dropout/Mul_1(1T㥛? <@9T㥛? <@AT㥛? <@IT㥛? <@a~އ?|j?i8?ɞ?g???Unknown
sHostCast"!sequential/dropout_1/dropout/Cast(1sh??|?:@9sh??|?:@Ash??|?:@Ish??|?:@a?	.?-i?iB?ΜȀ???Unknown
?HostMul"0gradient_tape/sequential/dropout_1/dropout/Mul_1(1??C??9@9??C??9@A??C??9@I??C??9@ai?㿄h?i??\M????Unknown
HostMul".gradient_tape/sequential/dropout_1/dropout/Mul(1?A`???7@9?A`???7@A?A`???7@I?A`???7@adg?|N?f?i??.??????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1V-???7@9V-???7@Aj?t?$3@Ij?t?$3@a?]?
qb?i?x9?????Unknown
dHostDataset"Iterator::Model(1j?t??[@9j?t??[@A1?ZD2@I1?ZD2@a?4a??Fa?i???E????Unknown
m HostSoftmax"sequential/dense/Softmax(1
ףp=*0@9
ףp=*0@A
ףp=*0@I
ףp=*0@a{??!Ǔ^?i?ǫӏ????Unknown
?!HostResourceApplyAdam"$Adam/Adam/update_5/ResourceApplyAdam(1J+?/@9J+?/@AJ+?/@IJ+?/@ai#?7g]?iɦ?[C????Unknown
`"HostGatherV2"
GatherV2_1(1J+??.@9J+??.@AJ+??.@IJ+??.@a??Ev?*]?i????????Unknown
i#HostWriteSummary"WriteSummary(1Zd;?O?(@9Zd;?O?(@AZd;?O?(@IZd;?O?(@a?x?uW?iͅz+????Unknown?
?$HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1???x??&@9???x??&@A???x??&@I???x??&@a?1?U?i??g???Unknown
q%HostCast"sequential/dropout/dropout/Cast(1-????&@9-????&@A-????&@I-????&@a{w?^mU?i?2?r!???Unknown
?&HostMul"Lgradient_tape/categorical_crossentropy/softmax_cross_entropy_with_logits/mul(1Zd;?OM&@9Zd;?OM&@AZd;?OM&@IZd;?OM&@a?y?`?U?i͕rW?+???Unknown
?'HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1?K7?A &@9?K7?A &@A?K7?A &@I?K7?A &@a2_?],?T?i}n?? 6???Unknown
?(HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1?????&@9?????&@A?????&@I?????&@a'?????T?iAnuP?@???Unknown
})HostReluGrad"'gradient_tape/sequential/conv0/ReluGrad(1q=
ףp%@9q=
ףp%@Aq=
ףp%@Iq=
ףp%@a?"GT?iD?vڱJ???Unknown
?*HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1??S??[$@9??S??[$@A??S??[$@I??S??[$@a?2?aAS?i?ЧgRT???Unknown
[+HostAddV2"Adam/add(1w??/?#@9w??/?#@Aw??/?#@Iw??/?#@a=??~?R?i| 
'?]???Unknown
l,HostIteratorGetNext"IteratorGetNext(1T㥛Ġ#@9T㥛Ġ#@AT㥛Ġ#@IT㥛Ġ#@aړN?Z?R?i?GmT?f???Unknown
?-HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1ףp=
 @9ףp=
 @Aףp=
 @Iףp=
 @al?ƭuoN?i??1?n???Unknown
v.HostAssignAddVariableOp"AssignAddVariableOp_2(1?? ?r?@9?? ?r?@A?? ?r?@I?? ?r?@a??K?-N?i??k?&v???Unknown
g/HostStridedSlice"strided_slice(1??(\??@9??(\??@A??(\??@I??(\??@a'?dÆ%L?i???/}???Unknown
o0HostMul"sequential/dropout/dropout/Mul(11?Zd@91?Zd@A1?Zd@I1?Zd@a???.?I?iqN??????Unknown
x1HostDataset"#Iterator::Model::ParallelMapV2::Zip(1???x??Y@9???x??Y@A???x??@I???x??@all$?5I?i??F?????Unknown
?2HostReadVariableOp"&sequential/conv0/Conv2D/ReadVariableOp(1sh??|?@9sh??|?@Ash??|?@Ish??|?@a+]?G?i?]???????Unknown
t3HostReadVariableOp"Adam/Cast/ReadVariableOp(1}?5^?I@9}?5^?I@A}?5^?I@I}?5^?I@a?=`???F?i??}??????Unknown
?4HostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1}?5^??@9}?5^??@A}?5^??@I}?5^??@aLK? u?E?i???????Unknown
Z5HostArgMax"ArgMax(1?A`?Т@9?A`?Т@A?A`?Т@I?A`?Т@a5?+??hE?i?$?;k????Unknown
V6HostSum"Sum_2(1+??N@9+??N@A+??N@I+??N@a?2?@'D?i-YCu????Unknown
?7HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1X9??v>@9X9??v>@AX9??v>@IX9??v>@aRpQ?%C?iI?Wc>????Unknown
\8HostArgMax"ArgMax_1(1;?O???@9;?O???@A;?O???@I;?O???@a????B?i?????????Unknown
?9HostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1?l????@9?l????@A?l????@I?l????@arc??A?iF?W?o????Unknown
?:HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1?G?z?@9?G?z?@A?G?z?@I?G?z?@a?\n?ҪA?i?CEڷ???Unknown
o;HostReadVariableOp"Adam/ReadVariableOp(1??x?&1@9??x?&1@A??x?&1@I??x?&1@a?V'?4A?i??o'????Unknown
?<HostTile";gradient_tape/categorical_crossentropy/weighted_loss/Tile_1(1?????K@9?????K@A?????K@I?????K@a??v]?[@?iKwp^>????Unknown
?=HostReadVariableOp"'sequential/conv0/BiasAdd/ReadVariableOp(1?V-@9?V-@A?V-@I?V-@a$??f?>@?iA7?	N????Unknown
{>HostSum"*categorical_crossentropy/weighted_loss/Sum(1??(\??@9??(\??@A??(\??@I??(\??@a???????ib?g?D????Unknown
e?Host
LogicalAnd"
LogicalAnd(1L7?A`?@9L7?A`?@AL7?A`?@IL7?A`?@aRttF<?i???T?????Unknown?
t@HostAssignAddVariableOp"AssignAddVariableOp(1+???@9+???@A+???@I+???@aϑ??%1<?i?ٴyS????Unknown
`AHostDivNoNan"
div_no_nan(1?O??n@9?O??n@A?O??n@I?O??n@a???q?~;?ik
X?????Unknown
YBHostPow"Adam/Pow(1V-??@9V-??@AV-??@IV-??@al?[/?;?i??Hi%????Unknown
vCHostAssignAddVariableOp"AssignAddVariableOp_4(1T㥛? 	@9T㥛? 	@AT㥛? 	@IT㥛? 	@ax?f?7?i???????Unknown
[DHostPow"
Adam/Pow_1(1j?t?@9j?t?@Aj?t?@Ij?t?@aH???&?6?i?D?.?????Unknown
]EHostCast"Adam/Cast_1(1NbX9?@9NbX9?@ANbX9?@INbX9?@a????=k6?i????????Unknown
?FHostReadVariableOp"'sequential/conv1/BiasAdd/ReadVariableOp(1V-???@9V-???@AV-???@IV-???@aUgU?J?5?i?_ٿz????Unknown
uGHostReadVariableOp"div_no_nan/ReadVariableOp(1Zd;?O@9Zd;?O@AZd;?O@IZd;?O@a?o/(4?i?M???????Unknown
?HHostDivNoNan"Egradient_tape/categorical_crossentropy/weighted_loss/value/div_no_nan(1NbX9?@9NbX9?@ANbX9?@INbX9?@a&?v??1?i???5????Unknown
bIHostDivNoNan"div_no_nan_1(1sh??|?@9sh??|?@Ash??|?@Ish??|?@a".P0?i;?S??????Unknown
?JHostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1j?t?R@9j?t?R@AV-?? @IV-?? @a????+n/?i???6????Unknown
VKHostCast"Cast(1?Zd; @9?Zd; @A?Zd; @I?Zd; @aSfz?8?.?i`?? "????Unknown
XLHostEqual"Equal(1?p=
ף??9?p=
ף??A?p=
ף??I?p=
ף??a@=?җ?,?iD???????Unknown
vMHostReadVariableOp"Adam/Cast_3/ReadVariableOp(1D?l?????9D?l?????AD?l?????ID?l?????a??dH,?i4???????Unknown
~NHostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1NbX9???9NbX9???ANbX9???INbX9???a????%+?i?+-?d????Unknown
?OHostDivNoNan",categorical_crossentropy/weighted_loss/value(1?Zd;??9?Zd;??A?Zd;??I?Zd;??ae&d?q?)?i?H????Unknown
wPHostReadVariableOp"div_no_nan_1/ReadVariableOp(1?O??n??9?O??n??A?O??n??I?O??n??a1T?e??)?i-n???????Unknown
XQHostCast"Cast_2(1????????9????????A????????I????????av?WR&?iO????????Unknown
vRHostReadVariableOp"Adam/Cast_2/ReadVariableOp(1?E??????9?E??????A?E??????I?E??????a??Ta?%?i??IQY????Unknown
TSHostMul"Mul(1ffffff??9ffffff??Affffff??Iffffff??a??ڶ?/%?iGi?I?????Unknown
vTHostAssignAddVariableOp"AssignAddVariableOp_3(1m???????9m???????Am???????Im???????a?8~?$?i?L]??????Unknown
?UHostReadVariableOp"&sequential/conv1/Conv2D/ReadVariableOp(1??Q????9??Q????A??Q????I??Q????a?oq?rZ$?i?Ç1<????Unknown
vVHostAssignAddVariableOp"AssignAddVariableOp_1(1P??n???9P??n???AP??n???IP??n???a?f??%?!?i1M?ST????Unknown
?WHostCast"8categorical_crossentropy/weighted_loss/num_elements/Cast(1?MbX9??9?MbX9??A?MbX9??I?MbX9??a?RC?j<!?if?h????Unknown
XXHostCast"Cast_1(1ffffff??9ffffff??Affffff??Iffffff??a???{??if@lN????Unknown
yYHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1;?O??n??9;?O??n??A;?O??n??I;?O??n??ag??+C7?i?????????Unknown
wZHostReadVariableOp"div_no_nan/ReadVariableOp_1(1?t?V??9?t?V??A?t?V??I?t?V??a?? ?i??ؠ????Unknown
a[HostIdentity"Identity(1???x?&??9???x?&??A???x?&??I???x?&??a?yS????i?????????Unknown?2CPU