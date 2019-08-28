%
% superclass for all spectral discretizations based on pointwise sampling
%
% author: Martin F. Schiffner
% date: 2019-02-25
% modified: 2019-05-13
%
classdef spectral_points < discretizations.spectral

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)
        

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = spectral_points( settings_tx, settings_rx )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure cell array for settings_tx
            if ~iscell( settings_tx )
                settings_tx = { settings_tx };
            end

            % ensure cell array for settings_rx
            if ~iscell( settings_rx )
                settings_rx = { settings_rx };
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( settings_tx, settings_rx );

            %--------------------------------------------------------------
            % 2.) constructor of superclass
            %--------------------------------------------------------------
            objects@discretizations.spectral();
            objects = repmat( objects, size( settings_tx ) );

            %--------------------------------------------------------------
            % 3.) set independent and dependent properties
            %--------------------------------------------------------------
            for index_object = 1:numel( settings_tx )

                % ensure correct number of settings_tx{ index_object }
                if ~( isscalar( settings_tx{ index_object } ) || numel( settings_tx{ index_object } ) == numel( settings_rx{ index_object } ) )
                    errorStruct.message = sprintf( 'Number of elements in settings_tx{ %d } must be one or match settings_rx{ %d }!', index_object, index_object );
                    errorStruct.identifier = 'spectral_points:SizeMismatch';
                    error( errorStruct );
                end

                % ensure identical frequency axes
% TODO: stimmt so nicht! -> vergleich per setting notwendig
                temp = [ settings_tx{ index_object }.impulse_responses, settings_rx{ index_object }.impulse_responses ];
                if ~isequal( temp.axis )
                    errorStruct.message = sprintf( 'Excitation voltages and impulse responses in settings_tx( %d ) and settings_rx( %d ) must have identical axes!', index_object, index_object );
                    errorStruct.identifier = 'spectral_points:AxesMismatch';
                    error( errorStruct );
                end

                %----------------------------------------------------------
                % a) set independent properties
                %----------------------------------------------------------
                objects( index_object ).tx = settings_tx{ index_object };
                objects( index_object ).rx = settings_rx{ index_object };

                %----------------------------------------------------------
                % b) set dependent properties
                %----------------------------------------------------------
                

            end % for index_object = 1:numel( settings_tx )

        end % function objects = spectral_points( settings_tx, settings_rx )

	end % methods

end % classdef spectral_points < discretizations.spectral
