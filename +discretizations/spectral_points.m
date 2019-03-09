%
% superclass for all spectral discretizations
%
% author: Martin F. Schiffner
% date: 2019-02-25
% modified: 2019-03-09
%
classdef spectral_points < discretizations.spectral

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        tx ( :, : ) discretizations.spectral_points_tx
        rx ( :, : ) discretizations.spectral_points_rx

        % dependent properties
        set_f_unique ( 1, 1 ) discretizations.set_discrete_frequency

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
            % 3.) set independent properties
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
                    errorStruct.message     = 'options_spectral must be discretizations.options_spectral!';
                    errorStruct.identifier	= 'discretize:NoOptionsFrequency';
                    error( errorStruct );
                end

                % set independent properties
                objects( index_object ).tx = tx{ index_object };
                objects( index_object ).rx = rx{ index_object };

            end % for index_object = 1:numel( tx )

        end % function objects = spectral_points( tx, rx )

	end % methods

end % classdef spectral_points < discretizations.spectral