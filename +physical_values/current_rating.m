%
% superclass for all current ratings
%
% author: Martin F. Schiffner
% date: 2019-03-27
% modified: 2019-03-27
%
classdef current_rating < physical_values.physical_quantity_base

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = current_rating( varargin )

            %--------------------------------------------------------------
            % 1.) constructor of superclass
            %--------------------------------------------------------------
            objects@physical_values.physical_quantity_base( 4, varargin{ : } );

        end % function objects = current_rating( varargin )

	end % methods

end % classdef current_rating < physical_values.physical_quantity_base
