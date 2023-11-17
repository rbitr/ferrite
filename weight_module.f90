module precision_module
  implicit none
  integer, parameter :: wp = kind(1.0)
end module precision_module

! structs for reading weights, config information and state 
module weight_module
        use precision_module
        implicit none
        private wp

        type TransformerWeights
                real(kind=wp), allocatable :: word_embeddings(:,:)
                real(kind=wp), allocatable :: position_embeddings(:,:)
                real(kind=wp), allocatable :: emb_layer_norm_w(:)
                real(kind=wp), allocatable :: emb_layer_norm_b(:)
                real(kind=wp), allocatable :: wq(:,:,:)
                real(kind=wp), allocatable :: bq(:,:)
                real(kind=wp), allocatable :: wk(:,:,:)
                real(kind=wp), allocatable :: bk(:,:)
                real(kind=wp), allocatable :: wv(:,:,:)
                real(kind=wp), allocatable :: bv(:,:)
                real(kind=wp), allocatable :: wo(:,:,:)
                real(kind=wp), allocatable :: bo(:,:)
                real(kind=wp), allocatable :: sa_layer_norm_w(:,:)
                real(kind=wp), allocatable :: sa_layer_norm_b(:,:)
                real(kind=wp), allocatable :: w1(:,:,:)
                real(kind=wp), allocatable :: b1(:,:)
                real(kind=wp), allocatable :: w2(:,:,:)
                real(kind=wp), allocatable :: b2(:,:)
                real(kind=wp), allocatable :: out_layer_norm_w(:,:)
                real(kind=wp), allocatable :: out_layer_norm_b(:,:)
                real(kind=wp), allocatable :: linear(:,:)

        end type TransformerWeights

        type Config
                INTEGER :: emb_dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len
        end type Config

        type RunState

                real(kind=wp), allocatable :: att(:,:)
                real(kind=wp), allocatable :: key_cache(:,:,:)
                real(kind=wp), allocatable :: value_cache(:,:,:)
                real(kind=wp) :: times(5)

        end type RunState

end module weight_module



