%
% superclass for all signal matrices
%
% author: Martin F. Schiffner
% date: 2019-03-27
% modified: 2019-05-29
%
classdef signal_matrix

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = protected)

        % independent properties
        axis ( 1, 1 ) math.sequence_increasing
        samples ( :, : ) physical_values.physical_quantity

        % dependent properties
        N_signals ( 1, 1 ) { mustBeInteger, mustBePositive } = 1

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = signal_matrix( axes, samples )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % return for zero arguments
            if nargin == 0
                return;
            end

            % ensure class math.sequence_increasing
            if ~isa( axes, 'math.sequence_increasing' )
                errorStruct.message     = 'axes must be math.sequence_increasing!';
                errorStruct.identifier	= 'signal_matrix:NoIncreasingSequence';
                error( errorStruct );
            end

            % ensure cell array for samples
            if ~iscell( samples )
                samples = { samples };
            end

            % single axes / multiple samples
            if isscalar( axes ) && ~isscalar( samples )
                axes = repmat( axes, size( samples ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( axes, samples );

            %--------------------------------------------------------------
            % 2.) create signal matrices
            %--------------------------------------------------------------
            % repeat default signal matrix
            objects = repmat( objects, size( axes ) );

            % iterate signal matrices
            for index_object = 1:numel( objects )

                % ensure sample matrix
                if ~ismatrix( samples{ index_object } )
                    errorStruct.message = sprintf( 'samples{ %d } must be a matrix!', index_object );
                    errorStruct.identifier = 'signal_matrix:NoMatrix';
                    error( errorStruct );
                end

                % ensure correct sizes
                if abs( axes( index_object ) ) ~= size( samples{ index_object }, 1 )
                    errorStruct.message = sprintf( 'Cardinality of axes( %d ) must match the size of samples{ %d } along the first dimension!', index_object, index_object );
                    errorStruct.identifier = 'signal_matrix:SizeMismatch';
                    error( errorStruct );
                end

                % set independent properties
                objects( index_object ).axis = axes( index_object );
                objects( index_object ).samples = samples{ index_object };

                % set dependent properties
                objects( index_object ).N_signals = size( samples{ index_object }, 2 );

            end % for index_object = 1:numel( objects )

        end % function objects = signal_matrix( axes, samples )

        %------------------------------------------------------------------
        % orthonormal discrete Fourier transform (DFT)
        %------------------------------------------------------------------
        function [ signal_matrices, N_dft, deltas ] = DFT( signal_matrices, intervals_t, intervals_f )
% TODO: generalize for arbitrary physical units
% TODO: generalize for complex-valued samples
% TODO: summarize multiple signals into signal matrix, if frequency axes are identical
            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class math.interval with physical units of time
            if ~( isa( intervals_t, 'math.interval' ) && isa( [ intervals_t.lb ], 'physical_values.time' ) )
                errorStruct.message = 'intervals_t must be math.interval whose bounds are physical_values.time!';
                errorStruct.identifier = 'DFT:NoTimeIntervals';
                error( errorStruct );
            end

            % ensure class math.interval with physical units of frequency
            if ~( isa( intervals_f, 'math.interval' ) && isa( [ intervals_f.lb ], 'physical_values.frequency' ) )
                errorStruct.message = 'intervals_f must be math.interval whose bounds are physical_values.frequency!';
                errorStruct.identifier = 'DFT:NoFrequencyIntervals';
                error( errorStruct );
            end

            % multiple signal_matrices / single time interval
            if ~isscalar( signal_matrices ) && isscalar( intervals_t )
                intervals_t = repmat( intervals_t, size( signal_matrices ) );
            end

            % multiple signal_matrices / single frequency interval
            if ~isscalar( signal_matrices ) && isscalar( intervals_f )
                intervals_f = repmat( intervals_f, size( signal_matrices ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( signal_matrices, intervals_t, intervals_f );

            %--------------------------------------------------------------
            % 2.) compute orthonormal discrete Fourier transforms
            %--------------------------------------------------------------
            % extract axes and numbers of samples
            axes = reshape( [ signal_matrices.axis ], size( signal_matrices ) );
            N_samples_signal = abs( axes );

            % ensure regular temporal samples
            if ~isa( axes, 'math.sequence_increasing_regular' )
                errorStruct.message = 'signal_matrices.axis must be regular!';
                errorStruct.identifier = 'DFT:IrregularAxis';
                error( errorStruct );
            end

            % extract deltas and lower bounds
            deltas = reshape( [ axes.delta ], size( signal_matrices ) );
            lbs_q_signal = reshape( [ axes.q_lb ], size( signal_matrices ) );

            % ensure class physical_values.time
            if ~isa( deltas, 'physical_values.time' )
                errorStruct.message = 'signal_matrices.axis must be a sequence of class physical_values.time!';
                errorStruct.identifier = 'DFT:NoTimes';
                error( errorStruct );
            end

            % quantize recording time intervals and determine lengths
            intervals_t_quantized = quantize( intervals_t, deltas );
            T_rec = abs( intervals_t_quantized );
            lbs_q = reshape( [ intervals_t_quantized.q_lb ], size( signal_matrices ) );
            ubs_q = reshape( [ intervals_t_quantized.q_ub ], size( signal_matrices ) );
            N_dft = double( ubs_q - lbs_q );

            % numbers of zeros to pad
            N_zeros_pad = N_dft - N_samples_signal;

            % ensure that numbers of samples do not exceed the order of the DFT
            if any( N_zeros_pad( : ) < 0 )
                errorStruct.message = sprintf( 'Number of signal samples exceeds order of DFT!' );
                errorStruct.identifier = 'DFT:IntervalMismatch';
                error( errorStruct );
            end

            % compute axes of relevant frequencies
            axes_f = discretize( intervals_f, 1 ./ T_rec );
            samples_shift = lbs_q_signal - lbs_q;

            % specify cell array for samples_dft
            samples_dft = cell( size( signal_matrices ) );

            % iterate signal matrices
            for index_object = 1:numel( signal_matrices )

                % ensure real-valued samples
                if ~isreal( signal_matrices( index_object ).samples )
                    errorStruct.message = sprintf( 'signal_matrices( %d ).samples must be real-valued!', index_object );
                    errorStruct.identifier = 'DFT:NoRealSamples';
                    error( errorStruct );
                end

                % specify relevant indices
                indices_relevant = double( axes_f( index_object ).q_lb:axes_f( index_object ).q_ub ) + 1;

                % zero-pad and shift samples
                samples_act = [ signal_matrices( index_object ).samples; zeros( N_zeros_pad( index_object ), signal_matrices( index_object ).N_signals ) ];
                samples_act = circshift( samples_act, samples_shift( index_object ), 1 );

                % compute and truncate DFT
                DFT_act = fft( samples_act, N_dft( index_object ), 1 ) / sqrt( N_dft( index_object ) );
                samples_dft{ index_object } = DFT_act( indices_relevant, : );

            end % for index_object = 1:numel( signal_matrices )

            %--------------------------------------------------------------
            % 3.) create signal matrices
            %--------------------------------------------------------------
            signal_matrices = discretizations.signal_matrix( axes_f, samples_dft );

        end % function [ signal_matrices, N_dft, deltas ] = DFT( signal_matrices, intervals_t, intervals_f )

        %------------------------------------------------------------------
        % Fourier transform
        %------------------------------------------------------------------
        function signal_matrices = fourier_transform( signal_matrices, intervals_t, intervals_f )

            %--------------------------------------------------------------
            % 1.) compute orthonormal discrete Fourier transforms (DFTs)
            %--------------------------------------------------------------
            [ signal_matrices, N_dft, deltas ] = DFT( signal_matrices, intervals_t, intervals_f );

            %--------------------------------------------------------------
            % 2.) compute Fourier transform samples
            %--------------------------------------------------------------
            for index_object = 1:numel( signal_matrices )
% TODO: correct scaling?
                signal_matrices( index_object ).samples = deltas( index_object ) * sqrt( N_dft( index_object ) ) * signal_matrices( index_object ).samples;

            end % for index_object = 1:numel( signal_matrices )

        end % function signal_matrices = fourier_transform( signal_matrices, intervals_t, intervals_f )

        %------------------------------------------------------------------
        % Fourier coefficients
        %------------------------------------------------------------------
        function signal_matrices = fourier_coefficients( signal_matrices, intervals_t, intervals_f )

            %--------------------------------------------------------------
            % 1.) compute orthonormal discrete Fourier transforms (DFTs)
            %--------------------------------------------------------------
            [ signal_matrices, N_dft, deltas ] = DFT( signal_matrices, intervals_t, intervals_f );

            %--------------------------------------------------------------
            % 2.) compute Fourier series coefficients
            %--------------------------------------------------------------
            for index_object = 1:numel( signal_matrices )

                signal_matrices( index_object ).samples = N_dft( index_object ) * signal_matrices( index_object ).samples;

            end % for index_object = 1:numel( signal_matrices )

        end % function signal_matrices = fourier_coefficients( signal_matrices, intervals_t, intervals_f )

        %------------------------------------------------------------------
        % time-domain signal
        %------------------------------------------------------------------
        function signal_matrices = signal( signal_matrices, lbs_q, delta )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure integer bounds
            mustBeNonempty( lbs_q );
            mustBeInteger( lbs_q );

            % ensure class physical_values.time
            if ~isa( delta, 'physical_values.time' )
                errorStruct.message = 'delta must be physical_values.time!';
                errorStruct.identifier = 'signal:NoTime';
                error( errorStruct );
            end

            % multiple signal_matrices / single lbs_q
            if ~isscalar( signal_matrices ) && isscalar( lbs_q )
                lbs_q = repmat( lbs_q, size( signal_matrices ) );
            end

            % multiple signal_matrices / single delta
            if ~isscalar( signal_matrices ) && isscalar( delta )
                delta = repmat( delta, size( signal_matrices ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( signal_matrices, lbs_q, delta );

            %--------------------------------------------------------------
            % 2.) compute time-domain signals
            %--------------------------------------------------------------
            % extract axes
            axes_f = reshape( [ signal_matrices.axis ], size( signal_matrices ) );
            N_samples_f = abs( axes_f );

            % ensure regular samples
            if ~isa( axes_f, 'math.sequence_increasing_regular' )
                errorStruct.message = 'signal_matrices.axis must be regular!';
                errorStruct.identifier = 'signal:IrregularAxis';
                error( errorStruct );
            end

            % extract deltas
            deltas = reshape( [ axes_f.delta ], size( signal_matrices ) );

            % ensure class physical_values.frequency
            if ~isa( deltas, 'physical_values.frequency' )
                errorStruct.message = 'signal_matrices.axis must be a sequence of class physical_values.frequency!';
                errorStruct.identifier = 'signal:NoFrequencies';
                error( errorStruct );
            end

            % compute time axes
            T_rec = 1 ./ deltas;
% TODO: noninteger?
            N_samples_t = T_rec ./ delta;
% TODO: N_samples_t odd?
            axes_t = math.sequence_increasing_regular( lbs_q, lbs_q + N_samples_t - 1, delta );
            index_shift = ceil( N_samples_t / 2 );

            % specify cell array for samples_td
            samples_td = cell( size( signal_matrices ) );

            % iterate signal matrices
            for index_object = 1:numel( signal_matrices )

                % zero-pad and shift samples
                samples_act = [ signal_matrices( index_object ).samples; zeros( index_shift( index_object ) - N_samples_f( index_object ), signal_matrices( index_object ).N_signals ) ];
                samples_act = circshift( samples_act, axes_f( index_object ).q_lb, 1 );

                % compute signal samples
                samples_td{ index_object } = N_samples_t * ifft( samples_act, N_samples_t, 1, 'symmetric' );

                % shift samples
                samples_td{ index_object } = circshift( samples_td{ index_object }, -axes_t( index_object ).q_lb, 1 );

            end % for index_object = 1:numel( signal_matrices )

            %--------------------------------------------------------------
            % 3.) create signal matrices
            %--------------------------------------------------------------
            signal_matrices = discretizations.signal_matrix( axes_t, samples_td );

        end % function signal_matrices = signal( signal_matrices, lbs_q, delta )

        %------------------------------------------------------------------
        % merge compatible signal matrices
        %------------------------------------------------------------------
        function signal_matrix = merge( signal_matrices )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % quick exit for single signal_matrices
            if isscalar( signal_matrices )
                signal_matrix = signal_matrices;
                return;
            end

            % ensure identical axes
            if ~isequal( signal_matrices.axis )
                errorStruct.message = 'All signal matrices must have identical axes!';
                errorStruct.identifier = 'merge:AxesMismatch';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) perform merging
            %--------------------------------------------------------------
            % extract common axis
            axis_mgd = signal_matrices( 1 ).axis;

            % concatenate samples along second dimension
            samples_mgd = { signal_matrices.samples };
            samples_mgd = cat( 2, samples_mgd{ : } );

            %--------------------------------------------------------------
            % 3.) create signal matrix
            %--------------------------------------------------------------
            signal_matrix = discretizations.signal_matrix( axis_mgd, samples_mgd );

        end % function signal_matrix = merge( signal_matrices )

        %------------------------------------------------------------------
        % element-wise multiplication (overload times method)
        %------------------------------------------------------------------
        function args_1 = times( args_1, args_2 )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure classes discretizations.signal_matrix
            if ~( isa( args_1, 'discretizations.signal_matrix' ) && isa( args_2, 'discretizations.signal_matrix' ) )
                errorStruct.message = 'args_1 and args_2 must be discretizations.signal_matrix!';
                errorStruct.identifier = 'times:NoSignalArrays';
                error( errorStruct );
            end

            % multiple args_1 / single args_2
            if ~isscalar( args_1 ) && isscalar( args_2 )
                args_2 = repmat( args_2, size( args_1 ) );
            end

            % single args_1 / multiple args_2
            if isscalar( args_1 ) && ~isscalar( args_2 )
                args_1 = repmat( args_1, size( args_2 ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( args_1, args_2 );

            %--------------------------------------------------------------
            % 2.) element-wise multiplications
            %--------------------------------------------------------------
            % iterate signal matrices
            for index_object = 1:numel( args_1 )

                % ensure identical axes
                if ~isequal( args_1( index_object ).axis.members, args_2( index_object ).axis.members )
                    errorStruct.message = sprintf( 'args_1( %d ) and args_2( %d ) must have identical members in their axes!', index_object, index_object );
                    errorStruct.identifier = 'times:DiverseAxes';
                    error( errorStruct );
                end

                % perform element-wise multiplication
                args_1( index_object ).samples = args_1( index_object ).samples .* args_2( index_object ).samples;

            end % for index_object = 1:numel( args_1 )

        end % function args_1 = times( args_1, args_2 )

        %------------------------------------------------------------------
        % subsample
        %------------------------------------------------------------------
        function signal_matrices = subsample( signal_matrices, indices_axes )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % return for single argument or empty indices_axes
            if nargin < 2 || isempty( indices_axes )
                return;
            end

            % ensure cell array for indices_axes
            if ~iscell( indices_axes )
                indices_axes = { indices_axes };
            end

            % multiple signal_matrices / single indices_axes
            if ~isscalar( signal_matrices ) && isscalar( indices_axes )
                indices_axes = repmat( indices_axes, size( signal_matrices ) );
            end

            % single signal_matrices / multiple indices_axes
            if isscalar( signal_matrices ) && ~isscalar( indices_axes )
                signal_matrices = repmat( signal_matrices, size( indices_axes ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( signal_matrices, indices_axes );

            %--------------------------------------------------------------
            % 2.) perform subsampling
            %--------------------------------------------------------------
            % subsample axes
            axes_sub = subsample( [ signal_matrices.axis ], indices_axes );

            % iterate signal matrices
            for index_object = 1:numel( signal_matrices )

                % assign subsampled axes
                signal_matrices( index_object ).axis = axes_sub( index_object );

                % subsample samples
                signal_matrices( index_object ).samples = signal_matrices( index_object ).samples( indices_axes{ index_object }, : );

            end % for index_object = 1:numel( signal_matrices )

        end % function signal_matrices = subsample( signal_matrices, indices_axes )

        %------------------------------------------------------------------
        % cut out submatrix
        %------------------------------------------------------------------
        function signal_matrices = cut_out( signal_matrices, lbs, ubs )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class discretizations.signal_matrix
            if ~isa( signal_matrices, 'discretizations.signal_matrix' )
                errorStruct.message = 'signal_matrices must be discretizations.signal_matrix!';
                errorStruct.identifier = 'cut_out:NoSignalMatrices';
                error( errorStruct );
            end

            % method cut_out in math.sequence_increasing ensures correct arguments

            %--------------------------------------------------------------
            % 2.) perform cut out
            %--------------------------------------------------------------
            % cut out axes
            axes = reshape( [ signal_matrices.axis ], size( signal_matrices ) );
            [ axes_cut, indicators ] = cut_out( axes, lbs, ubs );

            % ensure cell array for indicators
            if ~iscell( indicators )
                indicators = { indicators };
            end

            % iterate signal matrices
            for index_object = 1:numel( signal_matrices )

                % cut out samples
                signal_matrices( index_object ).axis = axes_cut( index_object );
                signal_matrices( index_object ).samples = signal_matrices( index_object ).samples( indicators{ index_object }, : );

            end % for index_object = 1:numel( signal_matrices )

        end % function signal_matrices = cut_out( signal_matrices, lbs, ubs )

        %------------------------------------------------------------------
        % return vector
        %------------------------------------------------------------------
        function y = return_vector( signal_matrices )

            % extract samples
            samples_cell = { signal_matrices.samples };

            % convert matrices into column vectors
            for index_object = 1:numel( signal_matrices )
                samples_cell{ index_object } = samples_cell{ index_object }( : );
            end

            % concatenate column vectors
            y = cat( 1, samples_cell{ : } );

        end % function y = return_vector( signal_matrices )

        %------------------------------------------------------------------
        % energy
        %------------------------------------------------------------------
        function energies = energy( signal_matrices )

            % compute energy in samples
            energies = reshape( cellfun( @( x ) norm( x( : ) )^2, { signal_matrices.samples } ), size( signal_matrices ) );

        end % function energies = energy( signal_matrices )

        %------------------------------------------------------------------
        % data volume
        %------------------------------------------------------------------
        function volumes = data_volume( signal_matrices )

            % initialize data volumes
            volumes = physical_values.byte( zeros( size( signal_matrices ) ) );

            % iterate signal matrices
            for index_object = 1:numel( signal_matrices )

                samples_act = signal_matrices( index_object ).samples;
                S = whos( 'samples_act' );
                volumes( index_object ) = physical_values.byte( S.bytes );

            end % for index_object = 1:numel( signal_matrices )

        end % function volumes = data_volume( signal_matrices )

        %------------------------------------------------------------------
        % 2-D line plot
        %------------------------------------------------------------------
        function hdl = show( signal_matrices )

            % initialize hdl with zeros
            hdl = zeros( size( signal_matrices ) );

            % iterate signal matrices
            for index_object = 1:numel( signal_matrices )

                % create new figure
                hdl( index_object ) = figure( index_object );

                % plot all samples
                if isreal( signal_matrices( index_object ).samples )
                    plot( double( signal_matrices( index_object ).axis.members ), double( signal_matrices( index_object ).samples.' ) );
                else
                    plot( double( signal_matrices( index_object ).axis.members ), double( abs( signal_matrices( index_object ).samples ).' ) );
                end

            end % for index_object = 1:numel( signal_matrices )

        end % function hdl = show( signal_matrices )

    end % methods

end % classdef signal_matrix
