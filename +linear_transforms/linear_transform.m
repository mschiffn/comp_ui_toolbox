%
% superclass for all linear transforms
% author: Martin Schiffner
% date: 2016-08-12
%
classdef linear_transform

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)
        N_coefficients
        N_lattice
        str_name
    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function LT = linear_transform( N_coefficients, N_lattice, str_name )

            % internal properties
            LT.N_coefficients	= N_coefficients;
            LT.N_lattice        = N_lattice;
            LT.str_name         = str_name;
        end

        %------------------------------------------------------------------
        % size of forward transform
        %------------------------------------------------------------------
        function y = size_transform( LT )

            % return size of forward transform
            y = [ LT.N_coefficients; LT.N_lattice ];
        end

        %------------------------------------------------------------------
        % forward transform
        %------------------------------------------------------------------
        function y = forward_transform( LT, x )

        end

        %------------------------------------------------------------------
        % adjoint transform
        %------------------------------------------------------------------
        function y = adjoint_transform( LT, x )

        end

        %------------------------------------------------------------------
        % transform operator
        %------------------------------------------------------------------
        function y = operator_transform( LT, x, mode )

            if mode == 0
                % return size of forward transform
                y = LT.size_transform();
            elseif mode == 1
                % forward transform
                y = LT.forward_transform( x );
            elseif mode == 2
                % adjoint transform
                y = LT.adjoint_transform( x );
            end % if mode == 0

            % return column vector
            y = y(:);
        end

    end % methods
    
end % classdef linear_transform