	&p?n??!@&p?n??!@!&p?n??!@	L??ǯM??L??ǯM??!L??ǯM??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:&p?n??!@7T??7???A?}"OR!@Y?(?r??rEagerKernelExecute 0*	NbX9Xl@2U
Iterator::Model::ParallelMapV2??????!X???l?G@)??????1X???l?G@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?????/??!&2L??C@)????B??1?m?,C@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?"LQ.???!?S5L-@)Swe???1?W?k@:Preprocessing2F
Iterator::Model??+ٱ??!o?jI??I@)?}͑??1y??V4@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?	?y?m?!???????)?	?y?m?1???????:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?????Z??!?g??MlH@)?_>Y1\m?1???4?I??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorm????|g?!f???:??)m????|g?1f???:??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapSB??^~??!`@g?Z<D@)?mr??S?11?F͙???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 1.5% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9K??ǯM??I???@ɞX@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	7T??7???7T??7???!7T??7???      ??!       "      ??!       *      ??!       2	?}"OR!@?}"OR!@!?}"OR!@:      ??!       B      ??!       J	?(?r???(?r??!?(?r??R      ??!       Z	?(?r???(?r??!?(?r??b      ??!       JCPU_ONLYYK??ǯM??b q???@ɞX@