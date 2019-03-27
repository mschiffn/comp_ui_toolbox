%
% superclass for all lengths
%
% author: Martin F. Schiffner
% date: 2019-02-11
% modified: 2019-03-27
%
classdef length < physical_values.physical_quantity_base

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = length( varargin )

            %--------------------------------------------------------------
            % 1.) constructor of superclass
            %--------------------------------------------------------------
            objects@physical_values.physical_quantity_base( 1, varargin{ : } );

        end % function objects = length( varargin )

	end % methods

end % classdef length < physical_values.physical_quantity_base
