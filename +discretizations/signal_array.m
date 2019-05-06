%
% superclass for all signal arrays
%
% author: Martin F. Schiffner
% date: 2019-03-27
% modified: 2019-04-30
%
classdef signal_array

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
        function objects = signal_array( axes, samples )

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
                errorStruct.identifier	= 'signal_array:NoIncreasingSequence';
                error( errorStruct );
            end

            % ensure cell array for samples
            if ~iscell( samples )
                samples = { samples };
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( axes, samples );

            %--------------------------------------------------------------
            % 2.) create signal arrays
            %--------------------------------------------------------------
            % repeat default signal array
            objects = repmat( objects, size( axes ) );

            % iterate signal arrays
            for index_object = 1:numel( objects )

                % ensure sample matrix
                if ~ismatrix( samples{ index_object } )
                    errorStruct.message = sprintf( 'samples{ %d } must be a matrix!', index_object );
                    errorStruct.identifier = 'signal_array:NoMatrix';
                    error( errorStruct );
                end

                % ensure correct sizes
                if abs( axes( index_object ) ) ~= size( samples{ index_object }, 2 )
                    errorStruct.message = sprintf( 'Cardinality of axes( %d ) must match the size of samples{ %d } along the second dimension!', index_object, index_object );
                    errorStruct.identifier = 'signal_array:SizeMismatch';
                    error( errorStruct );
                end

                % set independent properties
                objects( index_object ).axis = axes( index_object );
                objects( index_object ).samples = samples{ index_object };

                % set dependent properties
                objects( index_object ).N_signals = size( samples{ index_object }, 1 );

            end % for index_object = 1:numel( objects )

        end % function objects = signal_array( axes, samples )

        %------------------------------------------------------------------
        % orthonormal discrete Fourier transform (DFT)
        %------------------------------------------------------------------
        function [ signal_arrays, N_dft, deltas ] = DFT( signal_arrays, intervals_t, intervals_f )
% TODO: generalize for arbitrary physical units
% TODO: generalize for complex-valued samples
% TODO: summarize multiple signals into signal array, if frequency axes are identical
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

            % multiple signal_arrays / single time interval
            if ~isscalar( signal_arrays ) && isscalar( intervals_t )
                intervals_t = repmat( intervals_t, size( signal_arrays ) );
            end

            % multiple signal_arrays / single frequency interval
            if ~isscalar( signal_arrays ) && isscalar( intervals_f )
                intervals_f = repmat( intervals_f, size( signal_arrays ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( signal_arrays, intervals_t, intervals_f );

            %--------------------------------------------------------------
            % 2.) compute orthonormal discrete Fourier transforms
            %--------------------------------------------------------------
            % extract axes and numbers of samples
            axes = reshape( [ signal_arrays.axis ], size( signal_arrays ) );
            N_samples_signal = abs( axes );

            % ensure regular temporal samples
            if ~isa( axes, 'math.sequence_increasing_regular' )
                errorStruct.message = 'signal_arrays.axis must be regular!';
                errorStruct.identifier = 'DFT:IrregularAxis';
                error( errorStruct );
            end

            % extract deltas and lower bounds
            deltas = reshape( [ axes.delta ], size( signal_arrays ) );
            lbs_q_signal = reshape( [ axes.q_lb ], size( signal_arrays ) );

            % ensure class physical_values.time
            if ~isa( deltas, 'physical_values.time' )
                errorStruct.message = 'signal_arrays.axis must be a sequence of class physical_values.time!';
                errorStruct.identifier = 'DFT:NoTimes';
                error( errorStruct );
            end

            % quantize recording time intervals and determine lengths
            intervals_t_quantized = quantize( intervals_t, deltas );
            T_rec = abs( intervals_t_quantized );
            lbs_q = reshape( [ intervals_t_quantized.q_lb ], size( signal_arrays ) );
            ubs_q = reshape( [ intervals_t_quantized.q_ub ], size( signal_arrays ) );
            N_dft = double( ubs_q - lbs_q );

            % ensure that numbers of samples do not exceed the order of the DFT
            if any( N_samples_signal(:) > N_dft(:) )
                errorStruct.message = sprintf( 'Number of signal samples %d exceeds order of DFT %d!', N_samples_signal, N_dft );
                errorStruct.identifier = 'DFT:IntervalMismatch';
                error( errorStruct );
            end

            % compute axes of relevant frequencies
            axes_f = discretize( intervals_f, 1 ./ T_rec );
            samples_shift = lbs_q_signal - lbs_q;

            % specify cell array for samples_dft
            samples_dft = cell( size( signal_arrays ) );

            % iterate signal arrays
            for index_object = 1:numel( signal_arrays )

                % ensure real-valued samples
                if ~isreal( signal_arrays( index_object ).samples )
                    errorStruct.message = sprintf( 'signal_arrays( %d ).samples must be real-valued!', index_object );
                    errorStruct.identifier = 'DFT:NoRealSamples';
                    error( errorStruct );
                end

                % specify relevant indices
                indices_relevant = double( axes_f( index_object ).q_lb:axes_f( index_object ).q_ub );

                % zero-pad and shift samples
                samples_act = [ signal_arrays( index_object ).samples, zeros( signal_arrays( index_object ).N_signals, N_dft( index_object ) - N_samples_signal( index_object ) ) ];
                samples_act = circshift( samples_act, samples_shift( index_object ), 2 );

                % compute and truncate DFT
                DFT_act = fft( samples_act, N_dft( index_object ), 2 ) / sqrt( N_dft( index_object ) );
                samples_dft{ index_object } = DFT_act( :, indices_relevant );

            end % for index_object = 1:numel( signal_arrays )

            %--------------------------------------------------------------
            % 3.) create signal arrays
            %--------------------------------------------------------------
            signal_arrays = discretizations.signal_array( axes_f, samples_dft );

        end % function [ signal_arrays, N_dft, deltas ] = DFT( signal_arrays, intervals_t, intervals_f )

        %------------------------------------------------------------------
        % Fourier transform
        %------------------------------------------------------------------
        function signal_arrays = fourier_transform( signal_arrays, intervals_t, intervals_f )

            %--------------------------------------------------------------
            % 1.) compute orthonormal discrete Fourier transforms (DFTs)
            %--------------------------------------------------------------
            [ signal_arrays, N_dft, deltas ] = DFT( signal_arrays, intervals_t, intervals_f );

            %--------------------------------------------------------------
            % 2.) compute Fourier transform samples
            %--------------------------------------------------------------
            for index_object = 1:numel( signal_arrays )
% TODO: correct scaling?
                signal_arrays( index_object ).samples = deltas( index_object ) * sqrt( N_dft( index_object ) ) * signal_arrays( index_object ).samples;

            end % for index_object = 1:numel( signal_arrays )

        end % function signal_arrays = fourier_transform( signal_arrays, intervals_t, intervals_f )

        %------------------------------------------------------------------
        % Fourier coefficients
        %------------------------------------------------------------------
        function signal_arrays = fourier_coefficients( signal_arrays, intervals_t, intervals_f )

            %--------------------------------------------------------------
            % 1.) compute orthonormal discrete Fourier transforms (DFTs)
            %--------------------------------------------------------------
            [ signal_arrays, N_dft, deltas ] = DFT( signal_arrays, intervals_t, intervals_f );

            %--------------------------------------------------------------
            % 2.) compute Fourier series coefficients
            %--------------------------------------------------------------
            for index_object = 1:numel( signal_arrays )

                signal_arrays( index_object ).samples = N_dft( index_object ) * signal_arrays( index_object ).samples;

            end % for index_object = 1:numel( signal_arrays )

        end % function signal_arrays = fourier_coefficients( signal_arrays, intervals_t, intervals_f )

        %------------------------------------------------------------------
        % time-domain signal
        %------------------------------------------------------------------
        function signal_arrays = signal( signal_arrays, lbs_q, delta )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure integer bounds
            mustBeNonempty( lbs_q );
            mustBeInteger( lbs_q );

            % ensure class physical_values.time
            if ~isa( delta, 'physical_values.time' )
                errorStruct.message     = 'delta must be physical_values.time!';
                errorStruct.identifier	= 'signal:NoTime';
                error( errorStruct );
            end

            % multiple signal_arrays / single lbs_q
            if ~isscalar( signal_arrays ) && isscalar( lbs_q )
                lbs_q = repmat( lbs_q, size( signal_arrays ) );
            end

            % multiple signal_arrays / single delta
            if ~isscalar( signal_arrays ) && isscalar( delta )
                delta = repmat( delta, size( signal_arrays ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( signal_arrays, lbs_q, delta );

            %--------------------------------------------------------------
            % 2.) compute time-domain signals
            %--------------------------------------------------------------
            % extract axes
            axes = reshape( [ signal_arrays.axis ], size( signal_arrays ) );

            % ensure regular samples
            if ~isa( axes, 'math.sequence_increasing_regular' )
                errorStruct.message = 'signal_arrays.axis must be regular!';
                errorStruct.identifier = 'signal:IrregularAxis';
                error( errorStruct );
            end

            % extract deltas
            deltas = reshape( [ axes.delta ], size( signal_arrays ) );

            % ensure class physical_values.frequency
            if ~isa( deltas, 'physical_values.frequency' )
                errorStruct.message = 'signal_arrays.axis must be a sequence of class physical_values.frequency!';
                errorStruct.identifier = 'signal:NoFrequencies';
                error( errorStruct );
            end

            % compute time axes
            T_rec = 1 ./ deltas;
            N_samples_t = T_rec ./ delta;
% TODO: N_samples_t odd?
            axes_t = math.sequence_increasing_regular( lbs_q, lbs_q + N_samples_t - 1, delta );
            index_shift = ceil( N_samples_t / 2 );

            % specify cell array for samples_td
            samples_td = cell( size( signal_arrays ) );

            % iterate signal arrays
            for index_object = 1:numel( signal_arrays )

                % zero-pad and shift samples
                samples_act = [ signal_arrays( index_object ).samples, zeros( signal_arrays( index_object ).N_signals, index_shift( index_object ) - abs( signal_arrays( index_object ).axis ) ) ];
                samples_act = circshift( samples_act, axes( index_object ).q_lb, 2 );

                % compute signal samples
                samples_td{ index_object } = N_samples_t * ifft( samples_act, N_samples_t, 2, 'symmetric' );

            end % for index_object = 1:numel( signal_arrays )

            %--------------------------------------------------------------
            % 3.) create signal arrays
            %--------------------------------------------------------------
            signal_arrays = discretizations.signal_array( axes_t, samples_td );

        end % function signal_arrays = signal( signal_arrays, lbs_q, delta )

        %------------------------------------------------------------------
        % merge compatible signal arrays
        %------------------------------------------------------------------
        function signal_array = merge( signal_arrays )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % quick exit for single signal_arrays
            if isscalar( signal_arrays )
                signal_array = signal_arrays;
                return;
            end

            % ensure identical axes
            if ~isequal( signal_arrays.axis )
                errorStruct.message = 'All signal arrays must have identical axes!';
                errorStruct.identifier = 'merge:AxesMismatch';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) perform merging
            %--------------------------------------------------------------
            % extract common axis
            axis_mgd = signal_arrays( 1 ).axis;

            % concatenate samples along first dimension
            samples_mgd = { signal_arrays.samples };
            samples_mgd = cat( 1, samples_mgd{ : } );

            %--------------------------------------------------------------
            % 3.) create signal array
            %--------------------------------------------------------------
            signal_array = discretizations.signal_array( axis_mgd, samples_mgd );

        end % function signal_array = merge( signal_arrays )

        %------------------------------------------------------------------
        % element-wise multiplication (overload times method)
        %------------------------------------------------------------------
        function args_1 = times( args_1, args_2 )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure classes discretizations.signal_array
            if ~( isa( args_1, 'discretizations.signal_array' ) && isa( args_2, 'discretizations.signal_array' ) )
                errorStruct.message = 'args_1 and args_2 must be discretizations.signal_array!';
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
            % iterate signal arrays
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
        function signal_arrays = subsample( signal_arrays, indices_axes )

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

            % multiple signal_arrays / single indices_axes
            if ~isscalar( signal_arrays ) && isscalar( indices_axes )
                indices_axes = repmat( indices_axes, size( signal_arrays ) );
            end

            % single signal_arrays / multiple indices_axes
            if isscalar( signal_arrays ) && ~isscalar( indices_axes )
                signal_arrays = repmat( signal_arrays, size( indices_axes ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( signal_arrays, indices_axes );

            %--------------------------------------------------------------
            % 2.) perform subsampling
            %--------------------------------------------------------------
            % subsample axes
            axes_sub = subsample( [ signal_arrays.axis ], indices_axes );

            % iterate signal arrays
            for index_object = 1:numel( signal_arrays )

                % assign subsampled axes
                signal_arrays( index_object ).axis = axes_sub( index_object );

                % subsample samples
                signal_arrays( index_object ).samples = signal_arrays( index_object ).samples( :, indices_axes{ index_object } );

            end % for index_object = 1:numel( signal_arrays )

        end % function signal_arrays = subsample( signal_arrays, indices_axes )

        %------------------------------------------------------------------
        % energy
        %------------------------------------------------------------------
        function energies = energy( signal_arrays )

            % compute energy in samples
            energies = reshape( cellfun( @( x ) norm( x( : ) ), { signal_arrays.samples } ), size( signal_arrays ) );

        end % function energies = energy( signal_arrays )

        %------------------------------------------------------------------
        % data volume
        %------------------------------------------------------------------
        function volumes = data_volume( signal_arrays )

            % initialize data volumes
            volumes = physical_values.byte( zeros( size( signal_arrays ) ) );

            % iterate signal arrays
            for index_object = 1:numel( signal_arrays )

                samples_act = signal_arrays( index_object ).samples;
                S = whos( 'samples_act' );
                volumes( index_object ) = physical_values.byte( S.bytes );

            end % for index_object = 1:numel( signal_arrays )

        end % function volumes = data_volume( signal_arrays )

        %------------------------------------------------------------------
        % 2-D line plot
        %------------------------------------------------------------------
        function hdl = show( signal_arrays )

            % initialize hdl with zeros
            hdl = zeros( size( signal_arrays ) );

            % iterate signal arrays
            for index_object = 1:numel( signal_arrays )

                % create new figure
                hdl( index_object ) = figure( index_object );

                % plot all samples
                if isreal( signal_arrays( index_object ).samples )
                    plot( double( signal_arrays( index_object ).axis.members ), double( signal_arrays( index_object ).samples ) );
                else
                    plot( double( signal_arrays( index_object ).axis.members ), double( abs( signal_arrays( index_object ).samples ) ) );
                end

            end % for index_object = 1:numel( signal_arrays )

        end % function hdl = show( signal_arrays )

    end % methods

end % classdef signal_array
