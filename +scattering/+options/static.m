%
% superclass for all static scattering operator options
%
% author: Martin F. Schiffner
% date: 2019-07-09
% modified: 2019-10-22
%
classdef static

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        discretization ( 1, 1 ) scattering.options.discretization = scattering.options.discretization	% spatiospectral discretization
        materials ( 1, 1 ) scattering.options.material = scattering.options.material.compressibility	% material parameters

    end % properties

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function object = static( varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % check number of arguments
            if nargin == 0
                return;
            end

            %--------------------------------------------------------------
            % 2.) parse arguments
            %--------------------------------------------------------------
            for index_arg = 1:nargin

                switch class( varargin{ index_arg } )

                    %------------------------------------------------------
                    % spatiospectral discretization options
                    %------------------------------------------------------
                    case 'scattering.options.discretization'
                        object.discretization = varargin{ index_arg };

                    %------------------------------------------------------
                    % material parameters
                    %------------------------------------------------------
                    case 'scattering.options.material'
                        object.materials = varargin{ index_arg };

                    %------------------------------------------------------
                    % unknown
                    %------------------------------------------------------
                    otherwise
                        errorStruct.message = sprintf( 'Class of varargin{ %d } is unknown!', index_arg );
                        errorStruct.identifier = 'options:UnknownClass';
                        error( errorStruct );

                end % switch class( varargin{ index_arg } )

            end % for index_arg = 1:nargin

        end % function object = static( varargin )

	end % methods

end % classdef static
