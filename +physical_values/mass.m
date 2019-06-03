%
% superclass for all masses
%
% author: Martin F. Schiffner
% date: 2019-03-26
% modified: 2019-06-02
%
classdef mass < physical_values.physical_quantity_base

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = mass( varargin )

            %--------------------------------------------------------------
            % 1.) constructor of superclass
            %--------------------------------------------------------------
            objects@physical_values.physical_quantity_base( 2, varargin{ : } );

        end % function objects = mass( varargin )

	end % methods

end % classdef mass < physical_values.physical_quantity_base
