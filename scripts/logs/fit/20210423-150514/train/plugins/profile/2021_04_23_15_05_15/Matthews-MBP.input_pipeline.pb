	mr???!@mr???!@!mr???!@	?g????g???!?g???"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:mr???!@̵h?V??A?b.?!@Y??8?#??rEagerKernelExecute 0*	?S㥛do@2U
Iterator::Model::ParallelMapV2q???????!?<?jCG@)q???????1?<?jCG@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateMi?-???!??????B@)Eg?E(???1?d????A@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatF????!LkZ???!@)?1ZGU??1??+?9@:Preprocessing2F
Iterator::Model?\??J??!?thWI@)????`??1??????@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?sa????!?I񋗨H@)??Tkaz?1??r??I@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice? @??t?!?!???I??)? @??t?1?!???I??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor	?L?nh?!E?$B????)	?L?nh?1E?$B????:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap<1??PN??!???a
?B@)??q???U?1A?:????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 1.7% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?g???I?c??S?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	̵h?V??̵h?V??!̵h?V??      ??!       "      ??!       *      ??!       2	?b.?!@?b.?!@!?b.?!@:      ??!       B      ??!       J	??8?#????8?#??!??8?#??R      ??!       Z	??8?#????8?#??!??8?#??b      ??!       JCPU_ONLYY?g???b q?c??S?X@