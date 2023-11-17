module arg_parse
        implicit none

        type args
                real :: temperature
                character(:), allocatable :: model_file
                character(:), allocatable :: prompt
                character(:), allocatable :: tokenizer
                character(:), allocatable :: filename
                logical :: verbose
                integer :: n
                logical :: single_line, quiet, time
        end type args

        contains

                subroutine parse_args(arg_values)
                        type(args) :: arg_values
                        integer :: i, num_args
                        character(256) :: arg



                        !defaults 
                        arg_values%temperature = 0
                        arg_values%model_file = ""
                        arg_values%prompt = ""
                        arg_values%verbose = .false.
                        arg_values%n = 256
                        arg_values%tokenizer = "tokenizer.bin"
                        arg_values%single_line = .false.
                        arg_values%quiet = .false.
                        arg_values%filename = ""
                        arg_values%time = .false.

                        num_args = command_argument_count()

                        i = 1
                        do while (i <= num_args)
                                call get_command_argument(i, arg)
                                        select case (arg)
                                                case ('-m', '--model')
                                                ! path to model file
                                                call get_command_argument(i+1, arg)
                                                arg_values%model_file = trim(arg)
                                                i = i + 2
                                                case ('-p', '--prompt')
                                                ! prompt string
                                                call get_command_argument(i+1, arg)
                                                arg_values%prompt = trim(arg)
                                                i = i + 2
                                                case ('-s', '--tokenizer')
                                                ! path to custom tokenizer
                                                call get_command_argument(i+1, arg)
                                                arg_values%tokenizer = trim(arg)
                                                i = i + 2
                                                case ('-t', '--temperature')
                                                ! temperature scaling
                                                call get_command_argument(i+1, arg)
                                                read(arg,*) arg_values%temperature
                                                i = i + 2
                                                case ('-n', '--num_tokens')
                                                ! number of tokens to generate, including prompt
                                                call get_command_argument(i+1, arg)
                                                read(arg,*) arg_values%n
                                                i = i + 2
                                                case ('-f', '--filename')
                                                ! text file with a prompt on each line
                                                call get_command_argument(i+1, arg)
                                                arg_values%filename = trim(arg)

                                                i = i + 2
                                                case ('-v', '--verbose')
                                                ! print additional information
                                                arg_values%verbose = .true.
                                                i = i + 1
                                                case ('-1', '--single_line')
                                                ! print each element on single line
                                                arg_values%single_line = .true.
                                                i = i + 1
                                                case ('-q', '--quiet')
                                                        ! don't print embedding
                                                arg_values%quiet = .true.
                                                i = i + 1
                                                case ('--time')
                                                        ! display timings
                                                arg_values%time = .true.
                                                i = i + 1
                                                case default
                                                print *, 'Unrecognized option:', trim(arg)
                                                stop
                                                end select
                        end do

                        ! check for arguments


                end subroutine

end module arg_parse

program transformer

        use iso_c_binding
        use precision_module
        use weight_module
        use arg_parse
        use read_ggml, only: load_ggml
        implicit none

        type(TransformerWeights) :: weights
        type(Config) :: cfg

        integer(4) :: emb_dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len
        integer(4) :: itmp, msize

        type (args) :: arg_values
        character(:), allocatable :: prompt
        logical :: verbose, time

        real(kind=wp) :: score
        integer :: tok_len, max_len, n, p, l
        !integer :: vocab_size = 32000
        character(:), allocatable :: tmpstr
        character(:), dimension(:), allocatable :: vocab
        real(kind=wp),allocatable :: y(:), scores(:)
        integer, allocatable :: prompt_tokens(:)
        integer, allocatable :: vocab_len(:)
        integer, parameter :: max_prompt_len = 1024
        character(:), dimension(:), allocatable :: prompts
        character(len=max_prompt_len) :: temp_prompt
        integer :: tfid, ierr, num_lines
        real(kind=wp) :: t_start, t_end

        character(:), dimension(:), allocatable :: simple_tokens
        !integer, allocatable :: prompt_tokens


        call parse_args(arg_values)

        if (arg_values%prompt == "" .and. arg_values%filename == "") then
                !print *, arg_values%filename
                print *, "prompt required"
                stop
        end if

        verbose = arg_values%verbose
        time = arg_values%time

        msize = 0
        
        t_start = time_ms()
        
        call load_ggml(arg_values%model_file, weights, cfg, vocab, vocab_len, verbose)
        emb_dim = cfg%emb_dim
        hidden_dim = cfg%hidden_dim
        n_layers = cfg%n_layers
        n_heads = cfg%n_heads
        vocab_size = cfg%vocab_size
        seq_len = cfg%seq_len
        max_len = maxval(vocab_len)
        
        
        if (verbose) then
                !write(*,"(A,I0,A)") "Read ", vocab_size, " tokens"
                write(*,"(A,A)") "Token 4081 is ", vocab(4081)
        end if


        ! if there is a prompt, read the prompt and make a length 1 list
        ! if there is a file, read the lines into a list
        
        if (arg_values%prompt /= "") then
                allocate(character(len=max_prompt_len) ::  prompts(1))
                prompts(1) = arg_values%prompt
        
        else if (arg_values%filename /= "") then

                tfid = 5
                open(unit=tfid,file=arg_values%filename)
                ierr = 0
                num_lines = -1
                do while (ierr == 0)
                        num_lines = num_lines + 1
                        read(tfid,*,iostat=ierr) temp_prompt
                end do

                if (verbose) then
                        write(*,'(A,I0,A)') "Read ", num_lines, " lines"
                end if

                allocate(character(len=max_prompt_len) ::  prompts(num_lines))

                rewind(tfid)
                do p = 1,num_lines
                read(tfid, '(A)') prompts(p)
                end do
        
        end if
        
        t_end = time_ms()

        if (time) then
                print *, "Load time in seconds: ", (t_end-t_start)/1000
        end if  
        
        ! tokenize prompt
        !simple_tokens = simple_tokenize(arg_values%prompt)

        t_start = time_ms()
        do p=1,size(prompts)
        prompt_tokens = sp_tokenize(trim(prompts(p)))


        if (verbose) then 
        simple_tokens = simple_tokenize(trim(prompts(p)))
        do n=1,size(simple_tokens)
                print *, "simple token: ", simple_tokens(n)
                print *, "wordpiece tokens: ", encode_word(simple_tokens(n))
        end do

        print *, prompt_tokens

        end if

        !run through transformer
        y = dbert(prompt_tokens,weights,cfg)
        
        if (arg_values%quiet) then
                cycle
        end if 

        if (arg_values%single_line) then
                do n=1,emb_dim
                write (*,"(F10.5)") y(n)
                end do
        else
                print *, y
        end if

        end do
        t_end = time_ms()

        if (time) then
                print *, "Total inference time in seconds: ", (t_end-t_start)/1000
        end if


contains


        function layer_norm(x,w,b) result(xr)
              real(kind=wp) :: x(:,:), w(:), b(:)
              real(kind=wp) :: xr(size(x,1), size(x,2))
              real(kind=wp) :: xmean(size(x,1),size(x,2)), xvar(size(x,1),size(x,2))
              real(kind=wp) :: xn
              !print *, "A"
              xmean = spread(sum(x,dim=1)/size(x,1),1,size(x,1))
              xvar = spread(sum( (x-xmean)*(x-xmean),dim=1 ) / size(x,1), 1, size(x,1))
              xr = (x - xmean) / sqrt(xvar + 1e-12)
              !print *, "B"
              xr = xr*spread(w,2,size(x,2)) + spread(b,2,size(x,2))
        end function

        function softmax(x) result(y)
                real(kind=wp), intent(in) :: x(:,:)
                real(kind=wp) :: y(size(x,1),size(x,2))
                
                y = exp(x - spread(maxval(x,dim=1),1,size(x,1)))
                y = y / spread(sum(y,dim=1),1,size(x,1) )

        end function

        function attention(q,k,v) result(y)
                real(kind=wp), intent(in) :: q(:,:), k(:,:), v(:,:)
                real(kind=wp) :: y(size(q,1),size(q,2))
                real(kind=wp), allocatable :: y_int(:,:)
                y = matmul(v,(softmax(matmul(transpose(k),q) / sqrt(1.0*size(q,1)))))
        end function        

        function gelu(x) result(y)
                real(kind=wp), intent(in) :: x(:,:)
                real(kind=wp) :: y(size(x,1),size(x,2))
                y = 0.5 * x * (1 + tanh(sqrt(2 / 3.1415926536) * (x + 0.044715 * x**3)))
        end function

        function dbert(toks,w,c) result(y)
        integer, intent(in) :: toks(:)
        type(TransformerWeights) :: w
        type(Config) :: c
        real(kind=wp), allocatable :: y(:)
        integer :: i,j,l,h,nt,hsize
        real(kind=wp), allocatable :: x(:,:)

        real(kind=wp), allocatable :: q(:,:), k(:,:), v(:,:)
        real(kind=wp), allocatable :: qs(:,:,:), ks(:,:,:), vs(:,:,:)
        real(kind=wp), allocatable :: xb(:,:), attn_out(:,:), xbup(:,:)
        nt = size(toks)
        allocate(x(c%emb_dim, nt))
        allocate(y(c%emb_dim))
        allocate(xb(c%emb_dim, nt))
        allocate(attn_out(c%emb_dim,nt))
        allocate(xbup(c%hidden_dim,nt))

        hsize = c%emb_dim/c%n_heads

        do i=1,nt
        x(:,i) = w%word_embeddings(:,toks(i))
        x(:,i) = x(:,i) + w%position_embeddings(:,i)
        end do
        
        x = layer_norm(x,w%emb_layer_norm_w, w%emb_layer_norm_b)

        
        do l=1,c%n_layers
        
                q = matmul(transpose(w%wq(:,:,l)),x) + spread(w%bq(:,l),2,nt)
                k = matmul(transpose(w%wk(:,:,l)),x) + spread(w%bk(:,l),2,nt)
                v = matmul(transpose(w%wv(:,:,l)),x) + spread(w%bv(:,l),2,nt)
                
                ! split along embedding dim
                do h = 1,c%n_heads
                
                xb(((h-1)*hsize+1):(h*hsize),:) = attention( q(((h-1)*hsize+1):(h*hsize),:),&
                        &k(((h-1)*hsize+1):(h*hsize),:), v(((h-1)*hsize+1):(h*hsize),:))

                end do 

                xb = matmul(transpose(w%wo(:,:,l)),xb) + spread(w%bo(:,l),2,nt)
                xb = xb + x
                
                xb = layer_norm(xb,w%sa_layer_norm_w(:,l), w%sa_layer_norm_b(:,l))

                attn_out = xb

                xbup = matmul(transpose(w%w1(:,:,l)),xb) + spread(w%b1(:,l),2,nt)
                
                xbup = gelu(xbup)
                
                xb = matmul(transpose(w%w2(:,:,l)),xbup) + spread(w%b2(:,l),2,nt)
                
                xb = xb + attn_out

                x = layer_norm(xb,w%out_layer_norm_w(:,l), w%out_layer_norm_b(:,l))

        end do
        
        ! "pooling" average
        y = sum(x,dim=2) / size(x,2)

        ! linear
        y = matmul(transpose(w%linear), y)

        end function 
        
        function sp_tokenize(text) result(inds)
                character(len=*) :: text
                integer, allocatable :: inds(:)
                character(:), dimension(:), allocatable :: tokens, wpe
                integer :: m, n

                allocate(inds(1))

                inds(1) = 102 ! bos (1 added because 1 based indices)

                tokens = simple_tokenize(text)

                do m=1,size(tokens)
                        wpe = encode_word(tokens(m))
                        do n = 1,size(wpe)
                        inds = [inds, lookup(wpe(n),len_trim(wpe(n)))]
                        end do
                end do

                inds = [inds, 103]

        end function

        function lookup(s,l) result(ind)
                character(len=*) :: s
                integer :: l
                integer :: i, ind

                do i = 1,size(vocab)
                if (vocab(i) == s .and. vocab_len(i)==l) then
                        ind = i
                        return
                end if
                end do
                ind = -1
        end function

        function encode_word(word) result(tokens)
                character(len=*) :: word
                character(:), dimension(:), allocatable :: tokens
                integer :: i
                
                allocate(character(len=max_len) ::  tokens(0))

                do while(len_trim(word) > 0)
                        i = len_trim(word)
                        do while ( (i > 0) .and. (lookup(word(:i),i) <= 0))
                        i = i - 1
                        end do  
                        
                        if ( i == 0) then
                                deallocate(tokens)
                                tokens = ["UNK"]
                                return 
                        end if
                        tokens = [tokens, word(:i)]
                        !print *, tokens
                        word = word((i+1):)
                        if (len_trim(word) > 0) then
                                word = "##" // word
                        end if


                end do

        end function
        
        
        function simple_tokenize(text) result(tokens)
                character(len=*) :: text
                character(:), dimension(:), allocatable :: tokens
                character(:), allocatable :: ltext, allc
                character(len=max_len) :: next_token
                integer :: pos

                character(26), parameter :: alpha = 'abcdefghijklmnopqrstuvwxyz'
                character(35)  :: punct = '[!"#$%&\()*+,-./:;<=>?@\\^_`{|}~])x'
                character(10) :: numbers = '0123456789'
                
                ! is there another way to add the single quote?
                punct(35:35) = "'"
                !print *, punct
                allc = alpha // punct // numbers

                allocate(character(len=max_len) ::  tokens(0))

                ltext = to_lower(text)

                do while (len_trim(ltext) > 0)
                pos = 1
                
                next_token = ""
                
                
                
                do while(index(allc,ltext(pos:pos)) <= 0) 
                        pos = pos + 1
                end do

                ltext = ltext(pos:)

                pos = 1

                if (index(punct,ltext(pos:pos)) > 0 .and. pos <= len_trim(ltext)) then
                        !print *, index(punct,ltext(pos:pos))
                        next_token = ltext(pos:pos)
                        !if (verbose) then
                        !        print *, next_token
                        !end if 
                        tokens = [tokens, next_token]
                        ltext = ltext((pos+1):)
                        cycle 
                end if
                
                if (index(alpha,ltext(pos:pos)) > 0 .and. pos <= len_trim(ltext)) then !next char is alphabet
                
                do while(index(alpha,ltext(pos:pos)) > 0 .and. pos <= len_trim(ltext))
                        pos = pos + 1
                end do

                next_token = ltext(1:(pos-1))
                ltext = ltext(pos:)
                
                !if (verbose) then 
                        !print *, "control"
                        !print *, pos
                !        print *, next_token
                        !print *, ltext
                !end if
                
                ! fortran 2003?
                tokens = [tokens, next_token]

                else if (index(numbers,ltext(pos:pos)) > 0 .and. pos <= len_trim(ltext)) then ! next char is number
                do while(index(numbers,ltext(pos:pos)) > 0 .and. pos <= len_trim(ltext))
                        pos = pos + 1
                end do

                next_token = ltext(1:(pos-1))
                ltext = ltext(pos:)

                !if (verbose) then
                        !print *, "control"
                        !print *, pos
                !        print *, next_token
                        !print *, ltext
                !end if

                ! fortran 2003?
                tokens = [tokens, next_token]


                end if 

                end do

        
        end function

        !stackoverflow.com/questions/10759375/how-can-i-write-a-to-upper-or-to-lower-function-in-f90
        function to_lower (str) result (string)


        implicit None
        character(*), intent(in) :: str
        character(len(str))      :: string

        integer :: ic, i

        character(26), parameter :: cap = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        character(26), parameter :: low = 'abcdefghijklmnopqrstuvwxyz'

        string = str
        do i = 1, len_trim(str)
                ic = index(cap, str(i:i))
                if (ic > 0) string(i:i) = low(ic:ic)
        end do

        end function to_lower

        function time_ms() result(t_ms)
                real(kind=wp) :: t_ms
                integer(4) :: ms
                !call cpu_time(t_ms)
                call system_clock(ms)
                t_ms = real(ms)
        end function

end program
