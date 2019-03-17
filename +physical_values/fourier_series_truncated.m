%
% superclass for all truncated Fourier series
%
% author: Martin F. Schiffner
% date: 2019-02-22
% modified: 2019-02-22
%
classdef fourier_series_truncated

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        set_f ( 1, 1 ) discretizations.set_discrete_frequency	% set of discrete time instants
        coefficients %( 1, : ) physical_values.physical_value     % Fourier series coefficients

    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = fourier_series_truncated( sets_f, coefficients )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure coefficients is a cell array
            if ~iscell( coefficients )
                coefficients = { coefficients };
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( sets_f, coefficients );

            %--------------------------------------------------------------
            % 2.) create Fourier series
            %--------------------------------------------------------------
            % construct objects
            objects = repmat( objects, size( sets_f ) );

            % check and set independent properties
            for index_object = 1:numel( sets_f )

                % ensure row vectors with suitable numbers of components
                if ~( isrow( coefficients{ index_object } ) && numel( coefficients{ index_object } ) == abs( sets_f( index_object ) ) )
                    errorStruct.message     = sprintf( 'The content of coefficients{ %d } must be a row vector with %d components!', index_object, abs( sets_f( index_object ) ) );
                    errorStruct.identifier	= 'fourier_series_truncated:NoRowVector';
                    error( errorStruct );
                end

                % set independent properties
                objects( index_object ).set_f = sets_f( index_object );
                objects( index_object ).coefficients = coefficients{ index_object };

            end % for index_object = 1:numel( sets_f )

        end % function objects = fourier_series_truncated( sets_f, coefficients )

        %------------------------------------------------------------------
        % element-wise multiplication (overload times function)
        %------------------------------------------------------------------
        function objects_out = times( inputs_1, inputs_2 )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure coefficients is a cell array

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( inputs_1, inputs_2 );

            %--------------------------------------------------------------
            % 2.) create Fourier series
            %--------------------------------------------------------------
            % construct objects
            objects_out = inputs_1;

            % check and set independent properties
            for index_object = 1:numel( inputs_1 )

                % ensure identical frequency axes
                if ~isequal( inputs_1( index_object ).set_f.S, inputs_2( index_object ).set_f.S )
                    errorStruct.message     = sprintf( 'The content of coefficients{ %d } must be a row vector with %d components!', index_object, abs( sets_f( index_object ) ) );
                    errorStruct.identifier	= 'fourier_series_truncated:NoRowVector';
                    error( errorStruct );
                end

                % multiplication
                objects_out( index_object ).coefficients = inputs_1( index_object ).coefficients .* inputs_2( index_object ).samples;

            end % for index_object = 1:numel( inputs_1 )

        end % function objects_out = times( inputs_1, inputs_2 )

        %------------------------------------------------------------------
        % time-domain signal
        %------------------------------------------------------------------
        function signals_out = signal( objects_in, lbs_q, T_s )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure integer bounds
            if ~all( lbs_q( : ) == floor( lbs_q( : ) ) )
                errorStruct.message     = 'Boundary indices must be integers!';
                errorStruct.identifier	= 'signal:NoIntegerBounds';
                error( errorStruct );
            end

            % ensure class physical_values.time
            if ~isa( T_s, 'physical_values.time' )
                errorStruct.message     = 'T_s must be physical_values.time!';
                errorStruct.identifier	= 'signal:NoTime';
                error( errorStruct );
            end

            % multiple signals / single integer bound
            if ~isscalar( objects_in ) && isscalar( lbs_q )
                lbs_q = repmat( lbs_q, size( objects_in ) );
            end

            % multiple signals / single frequency interval
            if ~isscalar( objects_in ) && isscalar( T_s )
                T_s = repmat( T_s, size( objects_in ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( objects_in, lbs_q, T_s );

            %--------------------------------------------------------------
            % 2.) compute time-domain signals
            %--------------------------------------------------------------
            % initialize cell arrays
            sets_t = repmat( discretizations.set_discrete_time_regular( 0, 1, physical_values.time( 1 ) ), size( objects_in ) );
            samples = cell( size( objects_in ) );

            % iterate Fourier series
            for index_object = 1:numel( objects_in )

                % ensure class discretizations.set_discrete_frequency_regular
                if ~isa( objects_in( index_object ).set_f, 'discretizations.set_discrete_frequency_regular' )
                    errorStruct.message     = sprintf( 'objects_in( %d ).set_f must be discretizations.set_discrete_frequency_regular!', index_object );
                    errorStruct.identifier	= 'signal:NoRegularSampling';
                    error( errorStruct );
                end

                % compute time axis
                T_rec_act = 1 ./ objects_in( index_object ).set_f.F_s;
                N_samples_t = T_rec_act ./ T_s( index_object );
                sets_t( index_object ) = discretizations.set_discrete_time_regular( lbs_q( index_object ), lbs_q( index_object ) + N_samples_t - 1, T_s( index_object ) );

                % compute signal samples
                % TODO: N_samples_t odd?
                indices_relevant = double( objects_in( index_object ).set_f.q_lb:objects_in( index_object ).set_f.q_ub );
                index_shift = ceil( N_samples_t / 2 );
                dft = zeros( 1, index_shift );
                dft( indices_relevant ) = objects_in( index_object ).coefficients;
                samples{ index_object } = N_samples_t * ifft( dft, N_samples_t, 2, 'symmetric' );

            end % for index_object = 1:numel( objects_in )

            %--------------------------------------------------------------
            % 3.) create Fourier series coefficients
            %--------------------------------------------------------------
            signals_out = physical_values.signal( sets_t, samples );

        end % function signals_out = signal( objects_in, lbs_q, T_s )

        %------------------------------------------------------------------
        % 2-D line plot (overload plot function)
        %------------------------------------------------------------------
        function objects = plot( objects )

            % create new figure
            figure;

            % plot all signals in single figure
            plot( double( objects( 1 ).set_f.S ), abs( objects( 1 ).coefficients ) );
            hold on;
            for index_object = 2:numel( objects )
                plot( double( objects( index_object ).set_f.S ), abs( objects( index_object ).coefficients ) );
            end % for index_object = 2:numel( objects )
            hold off;

        end % function objects = plot( objects )

    end % methods

end
