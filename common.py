from tensorflow import flags
FLAGS = flags.FLAGS

flags.DEFINE_bool("gating_remove_diag", False,
                  "Remove diag for self gating")
flags.DEFINE_bool("lightvlad", False,
                  "Light or full NetVLAD")
flags.DEFINE_bool("vlagd", False,
                  "vlagd of vlad")
flags.DEFINE_integer("netvlad_cluster_size", 64,
                     "Number of units in the NetVLAD cluster layer.")

flags.DEFINE_bool("netvlad_relu", True, 'add ReLU to hidden layer')

flags.DEFINE_integer("netvlad_dimred", -1,
                     "NetVLAD output dimension reduction")

flags.DEFINE_integer("gatednetvlad_dimred", 1024,
                     "GatedNetVLAD output dimension reduction")

flags.DEFINE_bool("gating", False,
                  "Gating for NetVLAD")

flags.DEFINE_integer("hidden_size", 1024,
                     "size of hidden layer for BasicStatModel.")

flags.DEFINE_integer("netvlad_hidden_size", 1024,
                     "Number of units in the NetVLAD hidden layer.")

flags.DEFINE_integer("netvlad_hidden_size_video", 1024,
                     "Number of units in the NetVLAD video hidden layer.")

flags.DEFINE_integer("netvlad_hidden_size_audio", 64,
                     "Number of units in the NetVLAD audio hidden layer.")

flags.DEFINE_bool("netvlad_add_batch_norm", True,
                  "Adds batch normalization to the DBoF model.")

flags.DEFINE_integer("fv_cluster_size", 64,
                     "Number of units in the NetVLAD cluster layer.")

flags.DEFINE_integer("fv_hidden_size", 2048,
                     "Number of units in the NetVLAD hidden layer.")

flags.DEFINE_bool("fv_relu", True,
                  "ReLU after the NetFV hidden layer.")

flags.DEFINE_bool("fv_couple_weights", True,
                  "Coupling cluster weights or not")

flags.DEFINE_float("fv_coupling_factor", 0.01,
                   "Coupling factor")

flags.DEFINE_integer("nextvlad_cluster_size", 64, "Number of units in the NeXtVLAD cluster layer.")
flags.DEFINE_integer("nextvlad_hidden_size", 1024, "Number of units in the NeXtVLAD hidden layer.")

flags.DEFINE_integer("groups", 8, "number of groups in VLAD encoding")
flags.DEFINE_float("drop_rate", 0.5, "dropout ratio after VLAD encoding")
flags.DEFINE_integer("expansion", 2, "expansion ratio in Group NetVlad")
flags.DEFINE_integer("gating_reduction", 8, "reduction factor in se context gating")
flags.DEFINE_bool("enable_gate", True, "enable output gate")

flags.DEFINE_integer("mix_number", 3, "the number of gvlad models")
flags.DEFINE_float("cl_temperature", 2, "temperature in collaborative learning")
flags.DEFINE_float("cl_lambda", 1.0, "penalty factor of cl loss")

flags.DEFINE_integer("self_attention_n_head", 8,
                     "The multi head num.")
flags.DEFINE_integer("self_attention_n_layer", 1,
                     "layer num.")
flags.DEFINE_integer("self_attention_hidden_size", 1024,
                     "The number of units after attention cluster layer.")
flags.DEFINE_float("self_attention_hidden_dropout", 0.3,
                   "Dropout rate for clustering operation")
flags.DEFINE_float("self_attention_attention_dropout", 0.1,
                   "Dropout rate for Feed Forward operation")
flags.DEFINE_bool("self_avg_embed", True,
                     "layer num.")

flags.DEFINE_integer("video_cluster_size", 256,
                     "The size of video cluster.")
flags.DEFINE_integer("audio_cluster_size", 32,
                     "The size of audio cluster.")
flags.DEFINE_integer("filter_size", 2,
                     "The filter multiplier size for deep context gate.")
#flags.DEFINE_integer("hidden_size", 1024,
#                     "The number of units after attention cluster layer.")
flags.DEFINE_bool("shift_operation", True,
                  "True iff shift operation is on.")
flags.DEFINE_float("cluster_dropout", 0.7,
                   "Dropout rate for clustering operation")
flags.DEFINE_float("ff_dropout", 0.8,
                   "Dropout rate for Feed Forward operation")

