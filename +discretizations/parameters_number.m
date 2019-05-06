%
% superclass for all discretization parameters using numbers
%
% author: Martin F. Schiffner
% date: 2019-03-19
% modified: 2019-04-15
%
classdef parameters_number < discretizations.parameters

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        values ( 1, : ) double { mustBeInteger, mustBePositive } = [ 2, 4 ];

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = parameters_number( values )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % specify default values if no arguments
            if nargin == 0
                values = [ 2, 4 ];
            end

            % ensure cell array for values
            if ~iscell( values )
                values = { values };
            end

            %--------------------------------------------------------------
            % 2.) constructor of superclass
            %--------------------------------------------------------------
            objects@discretizations.parameters( values );

            %--------------------------------------------------------------
            % 3.) set independent properties
            %--------------------------------------------------------------
            for index_object = 1:numel( objects )
                objects( index_object ).values = values{ index_object };
            end

        end % function objects = parameters_number( values )

	end % methods

end % classdef parameters_number < discretizations.parameters
