	?Y?N?&@?Y?N?&@!?Y?N?&@	}????C??}????C??!}????C??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:?Y?N?&@o~?D???A,+MJA&@Y-?i??&??rEagerKernelExecute 0*	??(\??l@2U
Iterator::Model::ParallelMapV2%]3?f???!? M?l?G@)%]3?f???1? M?l?G@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?<??+??!???=AA@)?)?"??1??????@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?Or?Md??!???c'#@)㥛? ???1??? ??@:Preprocessing2F
Iterator::Model?i?:Ⱦ?!?u??TJ@)?f??e??1??@???@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??_#I??!R?B???@)??_#I??1R?B???@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??????!n??\b?G@)?F???x?1c<?4X@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor+N?f?m?!?U?z?X??)+N?f?m?1?U?z?X??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapްmQf???!???R?A@)? ݗ3?U?1?!????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 1.3% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9}????C??I?<<5??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	o~?D???o~?D???!o~?D???      ??!       "      ??!       *      ??!       2	,+MJA&@,+MJA&@!,+MJA&@:      ??!       B      ??!       J	-?i??&??-?i??&??!-?i??&??R      ??!       Z	-?i??&??-?i??&??!-?i??&??b      ??!       JCPU_ONLYY}????C??b q?<<5??X@