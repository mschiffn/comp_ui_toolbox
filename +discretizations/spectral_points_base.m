%
% superclass for all spectral discretizations
%
% author: Martin F. Schiffner
% date: 2019-03-09
% modified: 2019-03-09
%
classdef spectral_points_base

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        transfer_functions

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = spectral_points_base( transfer_functions )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure cell array for transfer_functions
            if ~iscell( transfer_functions )
                transfer_functions = { transfer_functions };
            end

            %--------------------------------------------------------------
            % 2.) create objects
            %--------------------------------------------------------------
            objects = repmat( objects, size( transfer_functions ) );

            %--------------------------------------------------------------
            % 3.) set independent properties
            %--------------------------------------------------------------
            for index_object = 1:numel( transfer_functions )

                % set independent properties
                objects( index_object ).transfer_functions = transfer_functions{ index_object };

            end % for index_object = 1:numel( transfer_functions )

        end % function objects = spectral_points_base( transfer_functions )

	end % methods

end % classdef spectral_points_base
