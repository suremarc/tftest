	ȔAո1@ȔAո1@!ȔAո1@	?l]?&????l]?&???!?l]?&???"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:ȔAո1@?Q?Q???A?y?S?d1@Y7¢"N'??rEagerKernelExecute 0*	?t?w@2U
Iterator::Model::ParallelMapV2L???!??!jT??bF@)L???!??1jT??bF@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?????!P?=??A@),??? ??1?N?o??;@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice/M??.??!b??(P?@)/M??.??1b??(P?@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?ן??N??!z m??F!@)"???ɩ??1V???>l@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??%?"??!?/?.?I@)V???n/??1:??N??@:Preprocessing2F
Iterator::Model???p???!?2?i?nH@)"5?b????1?G??_@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor1е/?g?!???????)1е/?g?1???????:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??8d???!??????A@)c?D(bQ?1]E?"j??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 1.3% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?l]?&???IM??d??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?Q?Q????Q?Q???!?Q?Q???      ??!       "      ??!       *      ??!       2	?y?S?d1@?y?S?d1@!?y?S?d1@:      ??!       B      ??!       J	7¢"N'??7¢"N'??!7¢"N'??R      ??!       Z	7¢"N'??7¢"N'??!7¢"N'??b      ??!       JCPU_ONLYY?l]?&???b qM??d??X@