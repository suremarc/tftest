	??? @??? @!??? @	???_e?????_e??!???_e??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:??? @;?*????A7S!? @Y4??O??rEagerKernelExecute 0*	?E????d@2U
Iterator::Model::ParallelMapV2"?uq??!?:?'?mG@)"?uq??1?:?'?mG@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate0????q??!?Y?f^?>@)c&Q/?4??1\??Z?;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatT1??c??!"?e??'@)?jQLސ?1?`???#@:Preprocessing2F
Iterator::Model??4F먶?!??M8zJ@)?L??~ބ?1?&??hb@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicec?ZB>?y?!=?ahE@)c?ZB>?y?1=?ahE@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???!??!H ????G@)??Χ?u?1?GU>0	@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorF~?,l?!B*&?xu @)F~?,l?1B*&?xu @:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?W?B?_??!?F.V??@)?ui??]?1??Tv|_??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 1.3% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9???_e??IM?ׁj?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	;?*????;?*????!;?*????      ??!       "      ??!       *      ??!       2	7S!? @7S!? @!7S!? @:      ??!       B      ??!       J	4??O??4??O??!4??O??R      ??!       Z	4??O??4??O??!4??O??b      ??!       JCPU_ONLYY???_e??b qM?ׁj?X@