	ū?m?O@ū?m?O@!ū?m?O@	??~?K?????~?K???!??~?K???"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:ū?m?O@???~??A????)?O@Y???????rEagerKernelExecute 0*	??n?pi@2U
Iterator::Model::ParallelMapV2????????!2?{Q?D@)????????12?{Q?D@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatenk?K??!Л7,T?@@)?P29?3??1?j???;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatB??	ܚ?!?tu0??)@)?)?:]??1f???$@:Preprocessing2F
Iterator::Model-???a??!?2H͟H@)??z?ю??14k{?rr@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?-?v????!*4?1߇@)?-?v????1*4?1߇@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???>??!hͷ2`?I@)?????m??1?g?Ym?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorta???u?!?m?a?@)ta???u?1?m?a?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap1E?4~???!<?b??(A@)m??)??b?1x?\?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??~?K???I??*ڡ?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???~?????~??!???~??      ??!       "      ??!       *      ??!       2	????)?O@????)?O@!????)?O@:      ??!       B      ??!       J	??????????????!???????R      ??!       Z	??????????????!???????b      ??!       JCPU_ONLYY??~?K???b q??*ڡ?X@