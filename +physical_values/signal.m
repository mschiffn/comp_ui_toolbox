%
% superclass for all temporal signals
%
% author: Martin F. Schiffner
% date: 2019-02-03
% modified: 2019-02-20
%
classdef signal

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        set_t ( 1, 1 ) discretizations.set_discrete_time	% set of discrete time instants
        samples ( 1, : ) physical_values.physical_value     % temporal samples of the signal

    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = signal( sets_t, samples )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure samples is a cell array
            if ~iscell( samples )
                samples = { samples };
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( sets_t, samples );

            %--------------------------------------------------------------
            % 2.) create signals
            %--------------------------------------------------------------
            % construct column vector of objects
            N_objects = numel( sets_t );
            objects = repmat( objects, size( sets_t ) );

            % check and set independent properties
            for index_object = 1:N_objects

                % ensure row vectors with suitable numbers of components
                if ~( isrow( samples{ index_object } ) && numel( samples{ index_object } ) == abs( sets_t( index_object ) ) )
                    errorStruct.message     = sprintf( 'The content of samples{ %d } must be a row vector with %d components!', index_object, abs( sets_t( index_object ) ) );
                    errorStruct.identifier	= 'signal:NoRowVector';
                    error( errorStruct );
                end

                % set independent properties
                objects( index_object ).set_t = sets_t( index_object );
                objects( index_object ).samples = samples{ index_object };

            end % for index_object = 1:N_objects

        end % function objects = signal( sets_t, samples )

        %------------------------------------------------------------------
        % discrete Fourier transform (orthonormal)
        %------------------------------------------------------------------
        function [sets_f, coefficients] = DFT( signals, intervals_t, intervals_f )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure classes physical_values.interval_time and physical_values.interval_frequency
            if ~( isa( intervals_t, 'physical_values.interval_time' ) && isa( intervals_f, 'physical_values.interval_frequency' ) )
                errorStruct.message     = 'intervals_t must be physical_values.interval_time and intervals_f must be physical_values.interval_frequency!';
                errorStruct.identifier	= 'fourier_coefficients:NoIntervals';
                error( errorStruct );
            end

            % multiple signals / single time interval
            if ~isscalar( signals ) && isscalar( intervals_t )
                intervals_t = repmat( intervals_t, size( signals ) );
            end

            % multiple signals / single frequency interval
            if ~isscalar( signals ) && isscalar( intervals_f )
                intervals_f = repmat( intervals_f, size( signals ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( signals, intervals_t, intervals_f );
            % assertion: signals, intervals_t, and intervals_f have equal size

            %--------------------------------------------------------------
            % 2.) compute discrete Fourier transform  (orthonormal)
            %--------------------------------------------------------------
            % initialize cell arrays
            sets_f = repmat( discretizations.set_discrete_frequency_regular( 0, 1, physical_values.frequency( 1 ) ), size( signals ) );
            coefficients = cell( size( signals ) );

            % check and set independent properties
            for index_object = 1:numel( signals )

                % ensure class discretizations.set_discrete_time_regular
                if ~isa( signals( index_object ).set_t, 'discretizations.set_discrete_time_regular' )
                    errorStruct.message     = sprintf( 'signals( %d ).set_t must be discretizations.set_discrete_time_regular!', index_object );
                    errorStruct.identifier	= 'fourier_coefficients:NoRegularSampling';
                    error( errorStruct );
                end

                % quantize recording time interval and determine duration
                interval_t_quantized = quantize( intervals_t( index_object ), signals( index_object ).set_t.T_s );
                T_rec_act = abs( interval_t_quantized );

                % ensure sufficient durations of intervals_t
                N_samples_interval = double( interval_t_quantized.q_ub - interval_t_quantized.q_lb );
                if N_samples_interval < abs( signals( index_object ).set_t )
                    errorStruct.message     = sprintf( 'signals( %d ).set_t must be included in intervals_t( %d )!', index_object, index_object );
                    errorStruct.identifier	= 'fourier_coefficients:IntervalMismatch';
                    error( errorStruct );
                end

                % compute axis of relevant frequencies
                % TODO: various types of discretization / parameter objects / regular vs irregular
                sets_f( index_object ) = discretize( intervals_f( index_object ), 1 ./ T_rec_act );
                indices_relevant = double( sets_f( index_object ).q_lb:sets_f( index_object ).q_ub );

                % compute discrete Fourier transform (orthonormal)
                coefficients_act = fft( double( signals( index_object ).samples ), N_samples_interval ) / sqrt( N_samples_interval );
                coefficients{ index_object } = coefficients_act( indices_relevant );

                % compute phase shift
                samples_shift = signals( index_object ).set_t.q_lb - interval_t_quantized.q_lb;
                if samples_shift ~= 0
                    axis_Omega = 2 * pi * indices_relevant / N_samples_interval;
                    phase_shift = exp( -1j * axis_Omega * double( samples_shift ) );
                    coefficients{ index_object } = coefficients{ index_object } .* phase_shift;
                end

            end % for index_object = 1:numel( signals )

        end % function [sets_f, coefficients] = DFT( signals, intervals_t, intervals_f )

        %------------------------------------------------------------------
        % Fourier transform
        %------------------------------------------------------------------
        function objects_out = fourier_transform( signals, intervals_t, intervals_f )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure classes physical_values.interval_time and physical_values.interval_frequency
            if ~( isa( intervals_t, 'physical_values.interval_time' ) && isa( intervals_f, 'physical_values.interval_frequency' ) )
                errorStruct.message     = 'intervals_t must be physical_values.interval_time and intervals_f must be physical_values.interval_frequency!';
                errorStruct.identifier	= 'fourier_transform:NoIntervals';
                error( errorStruct );
            end

            % multiple signals / single time interval
            if ~isscalar( signals ) && isscalar( intervals_t )
                intervals_t = repmat( intervals_t, size( signals ) );
            end

            % multiple signals / single frequency interval
            if ~isscalar( signals ) && isscalar( intervals_f )
                intervals_f = repmat( intervals_f, size( signals ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( signals, intervals_t, intervals_f );
            % assertion: signals, intervals_t, and intervals_f have equal size

            %--------------------------------------------------------------
            % 2.) compute Fourier transform samples
            %--------------------------------------------------------------
            % initialize cell arrays
            sets_f = repmat( discretizations.set_discrete_frequency_regular( 0, 1, physical_values.frequency( 1 ) ), size( signals ) );
            coefficients = cell( size( signals ) );

            % check and set independent properties
            for index_object = 1:numel( signals )

                % ensure class discretizations.set_discrete_time_regular
                if ~isa( signals( index_object ).set_t, 'discretizations.set_discrete_time_regular' )
                    errorStruct.message     = sprintf( 'signals( %d ).set_t must be discretizations.set_discrete_time_regular!', index_object );
                    errorStruct.identifier	= 'fourier_transform:NoRegularSampling';
                    error( errorStruct );
                end

                % quantize recording time interval and determine duration
                interval_t_quantized = quantize( intervals_t( index_object ), signals( index_object ).set_t.T_s );
                T_rec_act = abs( interval_t_quantized );

                % ensure that intervals_t includes the discrete times
                N_samples_interval = double( interval_t_quantized.q_ub - interval_t_quantized.q_lb );
                N_samples_signal = abs( signals( index_object ).set_t );
                if N_samples_signal > N_samples_interval
                    errorStruct.message     = sprintf( 'signals( %d ).set_t must be included in intervals_t( %d )!', index_object, index_object );
                    errorStruct.identifier	= 'fourier_transform:IntervalMismatch';
                    error( errorStruct );
                end

                % compute axis of relevant frequencies
                sets_f( index_object ) = discretize( intervals_f( index_object ), 1 ./ T_rec_act );
                indices_relevant = double( sets_f( index_object ).q_lb:sets_f( index_object ).q_ub );

                % compute and truncate Fourier series coefficients
                samples_shift = signals( index_object ).set_t.q_lb - interval_t_quantized.q_lb;
                samples = circshift( [ double( signals( index_object ).samples ), zeros( 1, N_samples_interval - N_samples_signal ) ], samples_shift, 2 );
                coefficients_act = double( signals( index_object ).set_t.T_s ) * fft( samples, N_samples_interval );
                coefficients{ index_object } = coefficients_act( indices_relevant );

            end % for index_object = 1:numel( signals )

            %--------------------------------------------------------------
            % 3.) create Fourier series samples
            %--------------------------------------------------------------
            objects_out = physical_values.fourier_transform( sets_f, coefficients );

        end % function objects_out = fourier_coefficients( signals, intervals_t, intervals_f )

        %------------------------------------------------------------------
        % Fourier coefficients
        %------------------------------------------------------------------
        function objects_out = fourier_coefficients( signals, intervals_t, intervals_f )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure classes physical_values.interval_time and physical_values.interval_frequency
            if ~( isa( intervals_t, 'physical_values.interval_time' ) && isa( intervals_f, 'physical_values.interval_frequency' ) )
                errorStruct.message     = 'intervals_t must be physical_values.interval_time and intervals_f must be physical_values.interval_frequency!';
                errorStruct.identifier	= 'fourier_coefficients:NoIntervals';
                error( errorStruct );
            end

            % multiple signals / single time interval
            if ~isscalar( signals ) && isscalar( intervals_t )
                intervals_t = repmat( intervals_t, size( signals ) );
            end

            % multiple signals / single frequency interval
            if ~isscalar( signals ) && isscalar( intervals_f )
                intervals_f = repmat( intervals_f, size( signals ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( signals, intervals_t, intervals_f );
            % assertion: signals, intervals_t, and intervals_f have equal size

            %--------------------------------------------------------------
            % 2.) compute Fourier series coefficients
            %--------------------------------------------------------------
            % initialize cell arrays
            sets_f = repmat( discretizations.set_discrete_frequency_regular( 0, 1, physical_values.frequency( 1 ) ), size( signals ) );
            coefficients = cell( size( signals ) );

            % check and set independent properties
            for index_object = 1:numel( signals )

                % ensure class discretizations.set_discrete_time_regular
                if ~isa( signals( index_object ).set_t, 'discretizations.set_discrete_time_regular' )
                    errorStruct.message     = sprintf( 'signals( %d ).set_t must be discretizations.set_discrete_time_regular!', index_object );
                    errorStruct.identifier	= 'fourier_coefficients:NoRegularSampling';
                    error( errorStruct );
                end

                % quantize recording time interval and determine duration
                interval_t_quantized = quantize( intervals_t( index_object ), signals( index_object ).set_t.T_s );
                T_rec_act = abs( interval_t_quantized );

                % ensure that intervals_t includes the discrete times
                N_samples_interval = double( interval_t_quantized.q_ub - interval_t_quantized.q_lb );
                N_samples_signal = abs( signals( index_object ).set_t );
                if N_samples_signal > N_samples_interval
                    errorStruct.message     = sprintf( 'signals( %d ).set_t must be included in intervals_t( %d )!', index_object, index_object );
                    errorStruct.identifier	= 'fourier_coefficients:IntervalMismatch';
                    error( errorStruct );
                end

                % compute axis of relevant frequencies
                sets_f( index_object ) = discretize( intervals_f( index_object ), 1 ./ T_rec_act );
                indices_relevant = double( sets_f( index_object ).q_lb:sets_f( index_object ).q_ub );

                % compute and truncate Fourier series coefficients
                samples_shift = signals( index_object ).set_t.q_lb - interval_t_quantized.q_lb;
                samples = circshift( [ double( signals( index_object ).samples ), zeros( 1, N_samples_interval - N_samples_signal ) ], samples_shift, 2 );
                coefficients_act = sqrt( N_samples_interval ) * fft( samples, N_samples_interval );
                coefficients{ index_object } = coefficients_act( indices_relevant );

            end % for index_object = 1:numel( signals )

            %--------------------------------------------------------------
            % 3.) create Fourier series coefficients
            %--------------------------------------------------------------
            objects_out = physical_values.fourier_series_truncated( sets_f, coefficients );

        end % function objects_out = fourier_coefficients( signals, intervals_t, intervals_f )

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

end
