%
% superclass for all spectral discretizations
%
% author: Martin F. Schiffner
% date: 2019-03-08
% modified: 2019-03-12
%
classdef spectral_points_tx < discretizations.spectral_points_base

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        excitation_voltages

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = spectral_points_tx( indices_active, transfer_functions, excitation_voltages )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure cell array for transfer_functions
            if ~iscell( transfer_functions )
                transfer_functions = { transfer_functions };
            end

            % ensure cell array for excitation_voltages
            if ~iscell( excitation_voltages )
                excitation_voltages = { excitation_voltages };
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( transfer_functions, excitation_voltages );

            %--------------------------------------------------------------
            % 2.) constructor of superclass
            %--------------------------------------------------------------
            objects@discretizations.spectral_points_base( indices_active, transfer_functions );

            %--------------------------------------------------------------
            % 3.) set independent properties
            %--------------------------------------------------------------
            for index_object = 1:numel( transfer_functions )

                % ensure identical frequency axes
                if ~isequal( objects( index_object ).transfer_functions.set_f, excitation_voltages{ index_object }.set_f )
                    errorStruct.message     = 'All sets of discrete frequencies must be identical!';
                    errorStruct.identifier	= 'spectral_points_tx:FrequencyMismatch';
                    error( errorStruct );
                end

                % set independent properties
                objects( index_object ).excitation_voltages = excitation_voltages{ index_object };

            end % for index_object = 1:numel( transfer_functions )

        end % function objects = spectral_points_tx( transfer_functions, excitation_voltages )

        %------------------------------------------------------------------
        % union
        %------------------------------------------------------------------
        function object_out = union( spectral_points_tx )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % return object if only one argument
            if numel( spectral_points_tx ) == 1
                object_out = spectral_points_tx;
                return;
            end
            % TODO: check matching dimensions

            %--------------------------------------------------------------
            % 2.) check arguments
            %--------------------------------------------------------------
            % extract sets of discrete frequencies
            sets_f = repmat( spectral_points_tx( 1 ).excitation_voltages( 1 ).set_f, size( spectral_points_tx ) );
            for index_object = 2:numel( spectral_points_tx )
                sets_f( index_object ) = spectral_points_tx( index_object ).excitation_voltages( 1 ).set_f;
            end

            N_samples_f = abs( sets_f );
            N_samples_f_cs = [ 0, cumsum( N_samples_f ) ];

            % get unique discrete frequencies
            [ set_f_unique, ia, ic ] = union( sets_f );
            N_samples_f_unique = numel( ia );

            % initialize cell arrays
            coefficients = cell( size( spectral_points_tx( 1 ).indices_active ) );
            samples = cell( size( spectral_points_tx( 1 ).indices_active ) );

            for index_active = 1:numel( spectral_points_tx( 1 ).indices_active )

                coefficients{ index_active } = zeros( 1, N_samples_f_unique );
                samples{ index_active } = zeros( 1, N_samples_f_unique );

                for index_f_unique = 1:N_samples_f_unique

                    index_object = sum( ( ia( index_f_unique ) - N_samples_f_cs ) > 0 );
                    index_f = ia( index_f_unique ) - N_samples_f_cs( index_object );

                    coefficients{ index_active }( index_f_unique ) = spectral_points_tx( index_object ).excitation_voltages( index_active ).coefficients( index_f );
                    samples{ index_active }( index_f_unique ) = spectral_points_tx( index_object ).transfer_functions( index_active ).samples( index_f );

                end % for index_f_unique = 1:N_samples_f_unique

            end % for index_active = 1:numel( spectral_points_tx( 1 ).indices_active )

            indices_active = spectral_points_tx( 1 ).indices_active;
            transfer_functions = physical_values.fourier_transform( repmat( set_f_unique, size( spectral_points_tx( 1 ).indices_active ) ), samples );
            excitation_voltages = physical_values.fourier_series_truncated( repmat( set_f_unique, size( spectral_points_tx( 1 ).indices_active ) ), coefficients );

            %--------------------------------------------------------------
            % 3.) create objects
            %--------------------------------------------------------------
            object_out = discretizations.spectral_points_tx( indices_active, transfer_functions, excitation_voltages );

        end % function object_out = union( spectral_points_tx )

	end % methods

end % classdef spectral_points_tx < discretizations.spectral_points_base
