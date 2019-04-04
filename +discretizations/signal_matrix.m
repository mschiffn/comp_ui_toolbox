%
% superclass for all signal matrices
%
% author: Martin F. Schiffner
% date: 2019-03-27
% modified: 2019-04-03
%
classdef signal_matrix

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        axis ( 1, 1 ) math.sequence_increasing
        samples %physical_values.physical_quantity

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

            % ensure cell array for samples
            if ~iscell( samples )
                samples = { samples };
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

                % size of sample matrix
                size_samples = size( samples{ index_object } );

                % ensure correct sizes
                if ~( abs( axes( index_object ) ) == size_samples( end ) )
                    errorStruct.message = sprintf( 'axes( %d ) must match samples{ %d }!', index_object, index_object );
                    errorStruct.identifier = 'signal_matrix:SizeMismatch';
                    error( errorStruct );
                end

                % set independent properties
                objects( index_object ).axis = axes( index_object );
                objects( index_object ).samples = samples{ index_object };

                % set dependent properties
                objects( index_object ).N_signals = prod( size_samples( 1:(end - 1) ) );

            end % for index_object = 1:numel( objects )

        end % function objects = signal_matrix( axes, samples )

        %------------------------------------------------------------------
        % data volume
        %------------------------------------------------------------------
        function volumes = data_volume( signal_matrices )

            % initialize data volumes
            volumes = physical_values.byte( zeros( size( signal_matrices ) ) );

            % iterate signal matrices
            for index_object = 1:numel( signal_matrices )

                samples_act = signal_matrices( index_object ).samples;
                S = whos('samples_act');
                volumes( index_object ) = physical_values.byte( S.bytes );

            end % for index_object = 1:numel( signal_matrices )

        end % function volumes = data_volume( signal_matrices )

        %------------------------------------------------------------------
        % orthonormal discrete Fourier transform
        %------------------------------------------------------------------
        function [ signal_matrices, N_dft, deltas ] = DFT( signal_matrices, intervals_t, intervals_f )
% TODO: generalize for arbitrary physical units
% TODO: generalize for complex-valued samples
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
            % extract axes
            axes = reshape( [ signal_matrices.axis ], size( signal_matrices ) );
            N_samples_signal = abs( axes );

            % ensure regular samples
            if ~isa( axes, 'math.sequence_increasing_regular' )
                errorStruct.message = 'signal_matrices.axis must be regular!';
                errorStruct.identifier = 'DFT:IrregularAxis';
                error( errorStruct );
            end

            % extract deltas
            deltas = reshape( [ axes.delta ], size( signal_matrices ) );

            % ensure class physical_values.time
            if ~isa( [ axes.members ], 'physical_values.time' )
                errorStruct.message = 'signal_matrices.axis must be a sequence of class physical_values.time!';
                errorStruct.identifier = 'DFT:NoTimes';
                error( errorStruct );
            end

            % quantize recording time intervals and determine lengths
            intervals_t_quantized = quantize( intervals_t, deltas );
            T_rec_act = abs( intervals_t_quantized );
            N_dft = double( [ intervals_t_quantized.q_ub ] - [ intervals_t_quantized.q_lb ] );

            % ensure that intervals_t includes the discrete times
            if N_samples_signal > N_dft
                errorStruct.message = sprintf( 'Number of signal samples %d exceeds order of DFT %d!', N_samples_signal, N_dft );
                errorStruct.identifier = 'DFT:IntervalMismatch';
                error( errorStruct );
            end

            % compute axes of relevant frequencies
            axes_f = discretize( intervals_f, 1 ./ T_rec_act );
            samples_shift = axes.q_lb - [ intervals_t_quantized.q_lb ];

            % initialize cell arrays
            samples_dft = cell( size( signal_matrices ) );

            % iterate signal matrices
            for index_object = 1:numel( signal_matrices )

                % TODO: only for real-valued signals!
                indices_relevant = double( axes_f( index_object ).q_lb:axes_f( index_object ).q_ub );

                % compute and truncate DFT                
                size_samples = size( signal_matrices( index_object ).samples );
                samples_act = cat( numel( size_samples ), signal_matrices( index_object ).samples, zeros( size_samples( 1:(end - 1) ), N_dft - N_samples_signal ) );
                samples_act = circshift( samples_act, samples_shift( index_object ), numel( size_samples ) );

                DFT_act = fft( samples_act, N_dft( index_object ), numel( size_samples ) ) / sqrt( N_dft( index_object ) );
                str_selector = repmat( {':'}, [ 1, numel( size_samples ) - 1 ] );
                samples_dft{ index_object } = DFT_act( str_selector{ : }, indices_relevant );

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
        % 2-D line plot (overload plot function)
        %------------------------------------------------------------------
        function objects = plot( objects )

            % create new figure
            figure;

            % plot all signals in single figure
            plot( double( objects( 1 ).set_t.S ), double( objects( 1 ).samples ) );
            hold on;
            for index_object = 2:numel( objects )
                plot( double( objects( index_object ).set_t.S ), double( objects( index_object ).samples ) );
            end % for index_object = 2:numel( objects )
            hold off;

        end % function objects = plot( objects )

    end % methods

end % classdef signal_matrix
