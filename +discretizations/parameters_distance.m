%
% superclass for all discretization parameters using distances
%
% author: Martin F. Schiffner
% date: 2019-03-19
% modified: 2019-03-19
%
classdef parameters_distance < discretizations.parameters

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        values ( 1, : ) double { mustBeReal, mustBePositive } = 76.2e-6 * ones(1, 3);

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = parameters_distance( values )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % specify default values if no arguments
            if nargin == 0
                values = 76.2e-6 * ones(1, 3);
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

        end % function objects = parameters_distance( values )

	end % methods

end % classdef parameters_distance < discretizations.parameters
