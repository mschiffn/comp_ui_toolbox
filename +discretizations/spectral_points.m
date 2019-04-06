%
% superclass for all spectral discretizations based on pointwise sampling
%
% author: Martin F. Schiffner
% date: 2019-02-25
% modified: 2019-03-17
%
classdef spectral_points < discretizations.spectral

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        tx ( :, : ) controls.setting_tx
        rx ( :, : ) controls.setting_rx

        % dependent properties
        tx_unique ( :, : ) controls.setting_tx
        indices_f_to_unique
        indices_unique_to_f
        indices_active

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = spectral_points( tx, rx )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure cell array for tx
            if ~iscell( tx )
                tx = { tx };
            end

            % ensure cell array for rx
            if ~iscell( rx )
                rx = { rx };
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( tx, rx );

            %--------------------------------------------------------------
            % 2.) constructor of superclass
            %--------------------------------------------------------------
            objects@discretizations.spectral();
            objects = repmat( objects, size( tx ) );

            %--------------------------------------------------------------
            % 3.) set independent and dependent properties
            %--------------------------------------------------------------
            for index_object = 1:numel( tx )

                % ensure identical frequency axes
%                 transfer_functions_tx = [ tx{ index_object }.transfer_functions ];
%                 transfer_functions_rx = [ rx{ index_object }.transfer_functions ];
% 
%                 if ~isequal( tx{ index_object }.transfer_functions.set_f, rx{ index_object }.transfer_functions.set_f )
%                     errorStruct.message     = 'All sets of discrete frequencies must be identical!';
%                     errorStruct.identifier	= 'spectral_points:FrequencyMismatch';
%                     error( errorStruct );
%                 end
                if ~( numel( tx{ index_object } ) == 1 || numel( tx{ index_object } ) == numel( rx{ index_object } ) )
                    errorStruct.message     = 'Number of elements in tx must be one or match rx!';
                    errorStruct.identifier	= 'spectral_points:SizeMismatch';
                    error( errorStruct );
                end

                % set independent properties
                objects( index_object ).tx = tx{ index_object };
                objects( index_object ).rx = rx{ index_object };

                % set dependent properties
                [ objects( index_object ).tx_unique, objects( index_object ).indices_unique_to_f, objects( index_object ).indices_f_to_unique ] = unique( objects( index_object ).tx );
                objects( index_object ).indices_active = unique_indices( objects( index_object ).rx );

            end % for index_object = 1:numel( tx )

        end % function objects = spectral_points( tx, rx )

	end % methods

end % classdef spectral_points < discretizations.spectral
