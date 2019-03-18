%
% superclass for all scattering operator options
%
% author: Martin F. Schiffner
% date: 2019-02-15
% modified: 2019-03-18
%
classdef options

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        discretization ( 1, 1 ) discretizations.options = discretizations.options % spatiospectral discretization
        spatial_aliasing

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function object = options( varargin )

            % check number of arguments
            if nargin ~= 1
                return;
            end

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class discretizations.method
            if ~isa( method, 'discretizations.method' ) || numel( method ) ~= 1
                errorStruct.message     = 'method must be a single discretizations.method!';
                errorStruct.identifier	= 'scattering_operator:NoSingleMethod';
                error( errorStruct );
            end
            % assertion: method is a single discretizations.method

            %--------------------------------------------------------------
            % 2.) set independent properties
            %--------------------------------------------------------------

        end % function object = options( varargin )

	end % methods

end % classdef options
