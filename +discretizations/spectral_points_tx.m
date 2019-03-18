%
% superclass for all spectral discretizations based on pointwise sampling
%
% author: Martin F. Schiffner
% date: 2019-03-08
% modified: 2019-03-17
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
        % unique values in array (overload unique function)
        %------------------------------------------------------------------
        function [ object_out, indices_unique_to_f, indices_f_to_unique ] = unique( spectral_points_tx )

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
            % 2.) extract transfer functions and excitation voltages for unique frequencies
            %--------------------------------------------------------------
            % extract sets of discrete frequencies
            sets_f = repmat( spectral_points_tx( 1 ).excitation_voltages( 1 ).set_f, size( spectral_points_tx ) );
            for index_object = 2:numel( spectral_points_tx )
                sets_f( index_object ) = spectral_points_tx( index_object ).excitation_voltages( 1 ).set_f;
            end

            % extract set of unique discrete frequencies
            [ set_f_unique, indices_unique_to_f, indices_f_to_unique ] = unique( sets_f );
            N_samples_f_unique = numel( indices_unique_to_f );

            % initialize cell arrays
            coefficients = cell( size( spectral_points_tx( 1 ).indices_active ) );
            samples = cell( size( spectral_points_tx( 1 ).indices_active ) );

            for index_active = 1:numel( spectral_points_tx( 1 ).indices_active )

                % initialize coefficients and samples with zeros
                coefficients{ index_active } = zeros( 1, N_samples_f_unique );
                samples{ index_active } = zeros( 1, N_samples_f_unique );

                for index_f_unique = 1:N_samples_f_unique

                    % map unique frequencies to object and frequency index
                    index_object = indices_unique_to_f( index_f_unique ).index_object;
                    index_f = indices_unique_to_f( index_f_unique ).index_f;

                    % extract coefficients and samples
                    coefficients{ index_active }( index_f_unique ) = spectral_points_tx( index_object ).excitation_voltages( index_active ).coefficients( index_f );
                    samples{ index_active }( index_f_unique ) = spectral_points_tx( index_object ).transfer_functions( index_active ).samples( index_f );

                end % for index_f_unique = 1:N_samples_f_unique

            end % for index_active = 1:numel( spectral_points_tx( 1 ).indices_active )

            % create transfer functions and excitation voltages
            indices_active = spectral_points_tx( 1 ).indices_active;
            sets_f_unique = repmat( set_f_unique, size( indices_active ) );
            transfer_functions = physical_values.fourier_transform( sets_f_unique, samples );
            excitation_voltages = physical_values.fourier_series_truncated( sets_f_unique, coefficients );

            %--------------------------------------------------------------
            % 3.) create objects
            %--------------------------------------------------------------
            object_out = discretizations.spectral_points_tx( indices_active, transfer_functions, excitation_voltages );

        end % function [ object_out, indices_unique_to_f, indices_f_to_unique ] = unique( spectral_points_tx )

	end % methods

end % classdef spectral_points_tx < discretizations.spectral_points_base
