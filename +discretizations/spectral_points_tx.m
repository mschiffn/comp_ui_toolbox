%
% superclass for all spectral discretizations
%
% author: Martin F. Schiffner
% date: 2019-03-08
% modified: 2019-03-08
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
        function objects = spectral_points_tx( transfer_functions, excitation_voltages )

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
            objects@discretizations.spectral_points_base( transfer_functions );

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

	end % methods

end % classdef spectral_points_tx < discretizations.spectral_points_base
